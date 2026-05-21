from __future__ import annotations

from pathlib import Path

from PIL import Image

from picca_search.domain import DenseVector, SparseVector
from picca_search.infrastructure.transformers_compat import (
    import_transformers_symbols,
    ort_provider_for_device,
    transformers_pretrained_kwargs,
)


WAON_SIGLIP_MODEL = "llm-jp/waon-siglip2-base-patch16-256"
LIGHT_SPLADE_MODEL = "bizreach-inc/light-splade-japanese-28M"


def validate_sparse_onnx_output_names(output_names: list[str]) -> None:
    if "logits" in output_names:
        return
    raise ValueError(
        "Invalid SPLADE ONNX model: expected output 'logits', "
        f"but found {output_names}. Re-export with "
        "`scripts/prepare_models.py --output-dir models/`."
    )


class WaonSiglipEncoder:
    def __init__(self, model_name: str = WAON_SIGLIP_MODEL, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        AutoModel, AutoProcessor = import_transformers_symbols("AutoModel", "AutoProcessor")
        model_path = Path(model_name)
        local_files_only = model_path.is_dir()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            **transformers_pretrained_kwargs(prefer_slow=True),
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

        self.text_max_length = int(self.model.config.text_config.max_position_embeddings)

    def encode_image(self, image_path: Path) -> DenseVector:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs = _prepare_transformer_inputs(inputs, device=self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return DenseVector.create(_normalized_values(self.torch, features))

    def encode_text(self, text: str) -> DenseVector:
        # SigLIPのテキスト特徴量はパディング戦略に敏感なため、固定長トークン化で安定した類似度を得る
        inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        inputs = _prepare_transformer_inputs(inputs, device=self.device)
        with self.torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return DenseVector.create(_normalized_values(self.torch, features))

    def encode_images(self, images: list[Image.Image]) -> list[DenseVector]:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = _prepare_transformer_inputs(inputs, device=self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
        normalized = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        batch_values = normalized.detach().cpu().tolist()
        return [DenseVector.create(row) for row in batch_values]

    def encode_images_from_paths(self, image_paths: list[Path]) -> list[DenseVector]:
        images: list[Image.Image] = []
        try:
            for image_path in image_paths:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
            return self.encode_images(images)
        finally:
            for image in images:
                image.close()


class SpladeJapaneseSparseEncoder:
    def __init__(
        self,
        model_name: str = LIGHT_SPLADE_MODEL,
        device: str | None = None,
        top_k: int = 256,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.top_k = top_k

        model_path = Path(model_name)
        if model_path.is_dir() and (model_path / "model.onnx").exists():
            self._uses_onnx = True
            _validate_sparse_onnx_model_dir(model_path)
            from optimum.onnxruntime import ORTModelForMaskedLM
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                **transformers_pretrained_kwargs(prefer_slow=False),
            )
            self.model = ORTModelForMaskedLM.from_pretrained(
                model_name,
                provider=ort_provider_for_device(
                    self.device,
                    require_accelerator=self.device == "cuda",
                ),
                local_files_only=True,
            )
        else:
            self._uses_onnx = False
            AutoModelForMaskedLM, AutoTokenizer = import_transformers_symbols(
                "AutoModelForMaskedLM",
                "AutoTokenizer",
            )
            local_files_only = model_path.is_dir()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                **transformers_pretrained_kwargs(prefer_slow=False),
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            ).to(self.device)
            self.model.eval()

        self.max_length = _resolve_sparse_max_length(
            tokenizer_max_length=getattr(self.tokenizer, "model_max_length", None),
            model_max_length=getattr(self.model.config, "max_position_embeddings", None),
        )

    def encode_text(self, text: str) -> SparseVector:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = _prepare_transformer_inputs(
            inputs,
            device=self.device,
            move_to_device=not self._uses_onnx,
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            weights = self.torch.log1p(self.torch.relu(logits))
            weights = weights * inputs["attention_mask"].unsqueeze(-1)
            vector = self.torch.max(weights, dim=1).values.squeeze(0)
            values, indices = self.torch.topk(vector, k=min(self.top_k, vector.shape[0]))
            non_zero = values > 0
            values = values[non_zero].detach().cpu().tolist()
            indices = indices[non_zero].detach().cpu().tolist()
        if len(indices) == 0:
            # SPLADE が完全にゼロを返す入力でも Qdrant へ空 sparse vector を渡さない。
            unknown_token_id = self.tokenizer.unk_token_id or 0
            return SparseVector.create([unknown_token_id], [1.0])
        return SparseVector.create(indices, values)

    def encode_texts(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = _prepare_transformer_inputs(
            inputs,
            device=self.device,
            move_to_device=not self._uses_onnx,
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            weights = self.torch.log1p(self.torch.relu(logits))
            weights = weights * inputs["attention_mask"].unsqueeze(-1)
            pooled = self.torch.max(weights, dim=1).values
        results: list[SparseVector] = []
        for i in range(len(texts)):
            vector = pooled[i]
            values, indices = self.torch.topk(vector, k=min(self.top_k, vector.shape[0]))
            non_zero = values > 0
            values = values[non_zero].detach().cpu().tolist()
            indices = indices[non_zero].detach().cpu().tolist()
            if len(indices) == 0:
                unknown_token_id = self.tokenizer.unk_token_id or 0
                results.append(SparseVector.create([unknown_token_id], [1.0]))
            else:
                results.append(SparseVector.create(indices, values))
        return results


def _normalized_values(torch, tensor) -> list[float]:
    normalized = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.squeeze(0).detach().cpu().tolist()


def _prepare_transformer_inputs(
    inputs,
    *,
    device: str,
    move_to_device: bool = True,
):
    prepared = inputs.to(device) if move_to_device and hasattr(inputs, "to") else inputs
    if isinstance(prepared, dict):
        return {key: value for key, value in prepared.items() if value is not None}
    return prepared


def _resolve_sparse_max_length(
    *, tokenizer_max_length: int | None, model_max_length: int | None
) -> int:
    if model_max_length is not None and model_max_length > 0:
        if tokenizer_max_length is None or tokenizer_max_length <= 0:
            return int(model_max_length)
        return int(min(tokenizer_max_length, model_max_length))

    if tokenizer_max_length is not None and tokenizer_max_length > 0:
        return int(tokenizer_max_length)

    raise ValueError("Unable to determine max token length for sparse encoder")


def _validate_sparse_onnx_model_dir(model_path: Path) -> None:
    import onnx

    model = onnx.load(str(model_path / "model.onnx"))
    validate_sparse_onnx_output_names([output.name for output in model.graph.output])
