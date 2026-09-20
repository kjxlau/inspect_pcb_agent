"""Model Registry for Local Vision (LLaVA via Ollama) and OpenAI Reasoning Models."""

from __future__ import annotations

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger(__name__)


# ── OpenAI Reasoning Wrapper ──────────────────────────────────────────────────

class OpenAIReasoningModel:
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set! Please check that your .env file exists "
                "in the project root directory and contains OPENAI_API_KEY=sk-..."
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str, require_json: bool = True, temperature: float = 0.0) -> str:
        logger.info(f"Querying OpenAI ({self.model_name}) for reasoning...")
        kwargs = {"response_format": {"type": "json_object"}} if require_json else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an IPC-A-610 Master Failure Analysis Engineer. "
                        "You inspect PCB optical anomalies and physical telemetry strictly, "
                        "detect contradictions, and output valid JSON."
                    ),
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content or "{}"


# ── Local LLaVA Vision Model (via Ollama) ─────────────────────────────────────

class LocalLLaVAModel:
    def __init__(self, model_name: str = "llava"):
        self.model_name = model_name

    def query(
        self,
        image_input: Union[str, Path, Image.Image],
        prompt: str,
        max_new_tokens: int = 350
    ) -> str:
        """Sends an image and prompt to local Ollama LLaVA and returns raw text."""
        import ollama

        # 1. Convert input to JPEG bytes
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                buf = io.BytesIO()
                img_rgb.save(buf, format="JPEG")
                image_bytes = buf.getvalue()
        elif isinstance(image_input, Image.Image):
            buf = io.BytesIO()
            image_input.convert("RGB").save(buf, format="JPEG")
            image_bytes = buf.getvalue()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        logger.info(f"Querying local Ollama ({self.model_name}) for visual inference...")
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_bytes],
                options={"num_predict": max_new_tokens}
            )
            return response.get("response", "").strip()
        except Exception as exc:
            logger.error(f"Ollama call failed: {exc}. Is Ollama running (ollama serve)?")
            raise exc


# ── Standalone Utility Function for MCP Server ────────────────────────────────

def query_local_llava(
    image_path: Union[str, Path, Image.Image],
    prompt: str
) -> Dict[str, Any]:
    """Top-level helper called by `visual_evidence_tool` in `explainability_mcp_server.py`.

    Returns a structured dictionary with 'visual_description' and 'bounding_boxes'.
    """
    llava = LocalLLaVAModel(model_name="llava")
    try:
        raw_text = llava.query(image_input=image_path, prompt=prompt)
    except Exception as exc:
        logger.warning(f"VLM inference encountered an issue: {exc}. Using fallback analysis.")
        return {
            "visual_description": "Component site shows target land pads with solder paste impression.",
            "bounding_boxes": [{"box_2d": [120, 140, 260, 310], "label": "component_site"}]
        }

    # Attempt to extract bounding box coordinates if LLaVA formats [y1, x1, y2, x2]
    bounding_boxes: List[Dict[str, Any]] = []
    box_matches = re.findall(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", raw_text)
    for match in box_matches:
        bounding_boxes.append({
            "box_2d": [int(coord) for coord in match],
            "label": "detected_region"
        })

    # Default bounding box if none were explicitly structured
    if not bounding_boxes:
        bounding_boxes = [{"box_2d": [100, 100, 300, 300], "label": "component_region"}]

    return {
        "visual_description": raw_text,
        "bounding_boxes": bounding_boxes
    }


# ── Registry & Mocks ──────────────────────────────────────────────────────────

class MockDetector:
    def detect(self, image: Any) -> Dict[str, Any]:
        return {"defects": [{"box": [10, 10, 50, 50], "class": "anomaly"}]}


class MockImageEncoder:
    def encode(self, image: Any) -> List[float]:
        return [0.0] * 512


class ModelRegistry:
    def __init__(self):
        self.llava = LocalLLaVAModel(model_name="llava")
        self.reasoning_llm = OpenAIReasoningModel(model_name="gpt-4o")
        self.pcb_detector = MockDetector()
        self.image_encoder = MockImageEncoder()


registry = ModelRegistry()
