import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from contextlib import asynccontextmanager
import asyncio
import time
from binoculars_llama import Binoculars
from typing import Any

load_dotenv()

score_lock = asyncio.Lock()
class BinocularsManager:
    def __init__(self):
        self.detector = None
        self.last_used = None
        self.lock = asyncio.Lock()
        self.unload_task = None
        self.params = None

    async def load(self):
        async with self.lock:
            # Pobierz parametry z env
            models_dir = os.environ.get("BINOCULARS_MODELS_DIR")
            observer_model_name = os.environ.get("BINOCULARS_OBSERVER_MODEL")
            performer_model_name = os.environ.get("BINOCULARS_PERFORMER_MODEL")
            if not models_dir or not observer_model_name or not performer_model_name:
                raise RuntimeError("Missing required environment variables: BINOCULARS_MODELS_DIR, BINOCULARS_OBSERVER_MODEL, BINOCULARS_PERFORMER_MODEL")
            max_token_observed = int(os.environ.get("BINOCULARS_MAX_TOKEN_OBSERVED", 512))
            n_gpu_layers = int(os.environ.get("BINOCULARS_N_GPU_LAYERS", 50))
            mode = os.environ.get("BINOCULARS_MODE", "low-fpr")
            observer_model_path = os.path.join(str(models_dir), str(observer_model_name))
            performer_model_path = os.path.join(str(models_dir), str(performer_model_name))
            params = (observer_model_path, performer_model_path, max_token_observed, n_gpu_layers, mode)
            if self.detector is None or self.params != params:
                self.detector = Binoculars(
                    observer_model_path=observer_model_path,
                    performer_model_path=performer_model_path,
                    max_token_observed=max_token_observed,
                    n_gpu_layers_model=n_gpu_layers,
                    mode=mode
                )
                self.params = params
            self.last_used = time.time()
            if self.unload_task is None or self.unload_task.done():
                self.unload_task = asyncio.create_task(self._unload_after_timeout())
            return self.detector

    async def _unload_after_timeout(self):
        while True:
            await asyncio.sleep(60)
            if self.last_used and (time.time() - self.last_used > 600):
                async with self.lock:
                    self.detector = None
                    self.params = None
                    self.last_used = None
                break

    async def get(self):
        async with self.lock:
            self.last_used = time.time()
            return self.detector

bino_manager = BinocularsManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/score")
async def score(payload: Any = Body(...)):
    async with score_lock:
        if isinstance(payload, dict) and "text" in payload:
            text = payload["text"]
        else:
            text = payload

        if not (isinstance(text, str) or (isinstance(text, list) and all(isinstance(x, str) for x in text))):
            return {"error": "Input must be a string or a list of strings."}
        detector = await bino_manager.load()
        result = detector.compute_score(text)
        return {"score": result}
