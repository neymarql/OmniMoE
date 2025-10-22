"""FastAPI server exposing Omni-Stack MoE generation endpoint."""
from __future__ import annotations

import base64
import io
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from OmniMoE.inference.inference_engine import OmniInferenceEngine


class GenerateRequest(BaseModel):
    prompts: List[str]
    images_base64: List[str]
    max_new_tokens: int = 128


class GenerateResponse(BaseModel):
    responses: List[str]


def decode_images(images_base64: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for payload in images_base64:
        data = base64.b64decode(payload)
        images.append(Image.open(io.BytesIO(data)).convert("RGB"))
    return images


def build_app(model_dir: str, config_path: str) -> FastAPI:
    engine = OmniInferenceEngine(model_dir=model_dir, config_path=config_path)
    app = FastAPI(title="Omni-Stack MoE Server", version="1.0")

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        images = decode_images(request.images_base64)
        if len(images) != len(request.prompts):
            raise ValueError("Number of images must match number of prompts")
        responses = engine.generate(request.prompts, images, request.max_new_tokens)
        return GenerateResponse(responses=responses)

    return app


__all__ = ["build_app"]
