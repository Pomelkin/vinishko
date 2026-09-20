from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

from .maxsim import maxsim_inbatch
from .torch_utils import get_torch_device


class BaseVisualRetrieverProcessor(ABC):
    query_prefix: ClassVar[str] = ""
    query_augmentation_token: str

    @abstractmethod
    def process_images(
        self, images: list[Image.Image]
    ) -> BatchFeature | BatchEncoding:
        pass

    @abstractmethod
    def process_texts(self, texts: list[str]) -> BatchFeature | BatchEncoding:
        pass

    def process_queries(
        self,
        texts: list[str] | None = None,
        queries: list[str] | None = None,
        max_length: int = 50,
        contexts: list[str] | None = None,
        suffix: str | None = None,
    ) -> BatchFeature | BatchEncoding:
        if queries is not None:
            texts = queries
        if texts is None:
            raise ValueError("No texts or queries provided.")
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts = [self.query_prefix + text + suffix for text in texts]
        return self.process_texts(texts=texts)

    @abstractmethod
    def score(
        self,
        qs: torch.Tensor | list[torch.Tensor],
        ps: torch.Tensor | list[torch.Tensor],
        device: str | torch.device | None = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_multi_vector(
        qs: torch.Tensor | list[torch.Tensor],
        ps: torch.Tensor | list[torch.Tensor],
        batch_size: int = 128,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        device = device or get_torch_device("auto")
        scores_list: list[torch.Tensor] = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            ).to(device)
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(maxsim_inbatch(qs_batch, ps_batch))
            scores_list.append(torch.cat(scores_batch, dim=1).cpu())
        return torch.cat(scores_list, dim=0).to(torch.float32)

    @abstractmethod
    def get_n_patches(
        self, image_size: tuple[int, int], *args, **kwargs
    ) -> tuple[int, int]:
        pass
