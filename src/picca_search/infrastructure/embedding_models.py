from __future__ import annotations

from pathlib import Path

from PIL import Image

from picca_search.domain import DenseVector, SparseVector
from picca_search.infrastructure.transformers_compat import import_transformers_symbols


WAON_SIGLIP_MODEL = "llm-jp/waon-siglip2-base-patch16-256"
LIGHT_SPLADE_MODEL = "bizreach-inc/light-splade-japanese-28M"


class WaonSiglipEncoder:
    def __init__(self, model_name: str = WAON_SIGLIP_MODEL, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        model_path = Path(model_name)
        if model_path.is_dir() and (model_path / "model.onnx").exists():
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                model_name, provider=_get_ort_provider(self.device)
            )
        else:
            AutoModel, AutoProcessor = import_transformers_symbols("AutoModel", "AutoProcessor")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

        self.text_max_length = int(self.model.config.text_config.max_position_embeddings)

    def encode_image(self, image_path: Path) -> DenseVector:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = outputs.image_embeds
        return DenseVector.create(_normalized_values(self.torch, features))

    def encode_text(self, text: str) -> DenseVector:
        # SigLIP text features are sensitive to padding strategy; fixed-length tokenization
        # produces stable image-text similarity, while variable-length padding does not.
        inputs = self.processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                features = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = outputs.text_embeds
        return DenseVector.create(_normalized_values(self.torch, features))

    def encode_images(self, images: list[Image.Image]) -> list[DenseVector]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = outputs.image_embeds
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
            from optimum.onnxruntime import ORTModelForMaskedLM
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = ORTModelForMaskedLM.from_pretrained(
                model_name, provider=_get_ort_provider(self.device)
            )
        else:
            AutoModelForMaskedLM, AutoTokenizer = import_transformers_symbols(
                "AutoModelForMaskedLM",
                "AutoTokenizer",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
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
        ).to(self.device)
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
        ).to(self.device)
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


def _get_ort_provider(device: str) -> str:
    if device == "cuda":
        return "CUDAExecutionProvider"
    if device == "mps":
        # MPS is not well-supported in ORT yet, fallback to CPU or try CoreML
        return "CPUExecutionProvider"
    return "CPUExecutionProvider"


def _normalized_values(torch, tensor) -> list[float]:
    normalized = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.squeeze(0).detach().cpu().tolist()


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
