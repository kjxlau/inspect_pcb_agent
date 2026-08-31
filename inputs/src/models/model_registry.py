import io
import logging
import ollama
from PIL import Image

logger = logging.getLogger(__name__)

class LocalLLaVAModel:
    """Vision-Language Model using local LLaVA via Ollama."""
    def __init__(self, model_name: str = "llava"):
        self.model_name = model_name

    def query(self, image: Image.Image, prompt: str, max_new_tokens: int = 300) -> str:
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        logger.info(f"Querying Ollama (Model: {self.model_name}) for visual description...")
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image_bytes],
            options={"num_predict": max_new_tokens}
        )
        return response.get('response', '')


class LocalReasoningModel:
    """Text-only Reasoning Model using local Llama via Ollama."""
    def __init__(self, model_name: str = "llama3.1"):
        self.model_name = model_name

    def query(self, prompt: str, require_json: bool = False) -> str:
        kwargs = {"format": "json"} if require_json else {}
        logger.info(f"Querying Ollama (Model: {self.model_name}) for reasoning...")
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": 0.1},
            **kwargs
        )
        return response.get('response', '{}')


# --- Mocks for your other systems ---
class MockDetector:
    def detect(self, image):
        return {"defects": [{"box": [10, 10, 50, 50], "class": "anomaly"}]}

class MockImageEncoder:
    def encode(self, image):
        return [0.0] * 512 # Dummy embedding vector

class MockCaseDB:
    def query(self, component):
        return {"ipc_standard": "IPC-A-610 Class 3"}

class MockMeasurementSystem:
    def get_data(self, board_id, component_ref):
        return {"resistance_ohms": 99999, "height_um": 0} # Example: open circuit


class ModelRegistry:
    def __init__(self):
        self.llava = LocalLLaVAModel(model_name="llava")
        self.reasoning_llm = LocalReasoningModel(model_name="llama3.1")
        self.pcb_detector = MockDetector()
        self.image_encoder = MockImageEncoder()
        self.case_db = MockCaseDB()
        self.measurement_system = MockMeasurementSystem()

registry = ModelRegistry()
