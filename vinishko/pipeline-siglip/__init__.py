"""Поиск по целому фото через SigLIP2, без SAM3 и VLM."""

from .pipeline import Pipeline, PipelineResult
from .steps.vis_searcher import VisSearcher

__all__ = ["Pipeline", "PipelineResult", "VisSearcher"]
