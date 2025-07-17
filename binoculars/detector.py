from typing import Union
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import copy

from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars(object):
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
        compile: bool = False,
        check_tokenizer_consistency: bool = True,
    ) -> None:
        if check_tokenizer_consistency:
            assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)
        torch.set_float32_matmul_precision("medium")
        self.change_mode(mode)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        ).eval()
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        ).eval()
        if compile:
            self.observer_model = torch.compile(self.observer_model)
            self.performer_model = torch.compile(self.performer_model)
        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        )
        return encodings

    @torch.inference_mode()
    def _get_observer_logits(
        self, encodings_obs: transformers.BatchEncoding
    ) -> torch.Tensor:
        return self.observer_model(**encodings_obs).logits

    @torch.inference_mode()
    def _get_performer_logits(
        self, encodings_perf: transformers.BatchEncoding
    ) -> torch.Tensor:
        """FIXME: ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?) when using gptqmodel and GPTQ quantized LLM.
        This is because Triton is tied to cuda:0 for gptqmodel.
        If we are willing to only use one GPU, it works."""
        return self.performer_model(**encodings_perf).logits

    def _get_logits(
        self,
        encodings_obs: transformers.BatchEncoding,
        encodings_perf: transformers.BatchEncoding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observer_future = self.executor.submit(self._get_observer_logits, encodings_obs)
        performer_logits = self._get_performer_logits(encodings_perf)
        observer_logits = observer_future.result()
        return observer_logits, performer_logits

    def compute_encodings_score(
        self, encodings: transformers.BatchEncoding
    ) -> np.ndarray:
        obs_device = self.observer_model.device
        perf_device = self.performer_model.device
        # NOTE: `BatchEncoding.to()` mutates `self`.
        encodings_obs = copy(encodings).to(obs_device, non_blocking=True)
        encodings_perf = copy(encodings).to(perf_device, non_blocking=True)
        observer_logits, performer_logits = self._get_logits(
            encodings_obs, encodings_perf
        )
        ppl = perplexity(encodings_perf, performer_logits)
        x_ppl = entropy(
            observer_logits,
            performer_logits.to(obs_device),
            encodings_obs,
            self.tokenizer.pad_token_id,
        )
        binoculars_scores = ppl / x_ppl
        del (
            encodings_obs,
            encodings_perf,
            observer_logits,
            performer_logits,
            ppl,
            x_ppl,
        )
        return binoculars_scores

    def compute_score(
        self, input_text: Union[list[str], str]
    ) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        binoculars_scores = self.compute_encodings_score(encodings)
        binoculars_scores = binoculars_scores.tolist()
        return (
            binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores
        )

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(
            binoculars_scores < self.threshold,
            "Most likely AI-generated",
            "Most likely human-generated",
        ).tolist()
        return pred