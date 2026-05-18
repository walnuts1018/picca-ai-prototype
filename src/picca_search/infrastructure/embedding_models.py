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

        AutoModel, AutoProcessor = import_transformers_symbols("AutoModel", "AutoProcessor")

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.text_max_length = int(self.model.config.text_config.max_position_embeddings)
        self.model.eval()

    def encode_image(self, image_path: Path) -> DenseVector:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            inputs = self.processor(images=rgb_image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)
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
            features = self.model.get_text_features(**inputs)
        return DenseVector.create(_normalized_values(self.torch, features))


class SpladeJapaneseSparseEncoder:
    def __init__(
        self,
        model_name: str = LIGHT_SPLADE_MODEL,
        device: str | None = None,
        top_k: int = 256,
    ) -> None:
        import torch

        AutoModelForMaskedLM, AutoTokenizer = import_transformers_symbols(
            "AutoModelForMaskedLM",
            "AutoTokenizer",
        )

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.top_k = top_k
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_text(self, text: str) -> SparseVector:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
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


def _normalized_values(torch, tensor) -> list[float]:
    normalized = tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return normalized.squeeze(0).detach().cpu().tolist()
