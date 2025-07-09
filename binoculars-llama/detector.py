from typing import Union
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
from .metrics import perplexity, entropy

BINOCULARS_ACCURACY_THRESHOLD = 1.2
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527

CONTROL_TOKENS = [
    '<|tool|>', '<|system|>', '<|user|>', '<|assistant|>', '<|im_start|>', '<|im_end|>'
]

def get_control_token_ids(tokenizer_or_model):
    token_ids = set()
    for t in CONTROL_TOKENS:
        try:
            tid = tokenizer_or_model.tokenize(t.encode("utf-8"))
            if isinstance(tid, list):
                token_ids.update(tid)
            elif isinstance(tid, int):
                token_ids.add(tid)
        except Exception:
            continue
    return set(token_ids)

class Binoculars:
    def __init__(self,
                 observer_model_path: str,
                 performer_model_path: str,
                 max_token_observed: int = 512,
                 n_gpu_layers_model: int = 50,
                 mode: str = "low-fpr"):
        if not os.path.exists(observer_model_path) or not os.path.exists(performer_model_path):
            raise FileNotFoundError("The path doesn't exist.")
        self.change_mode(mode)
        self.max_token_observed = max_token_observed

        self.observer = Llama(model_path=observer_model_path,
                              n_ctx=max_token_observed,
                              logits_all=True,
                              verbose=False,
                              n_gpu_layers=int(n_gpu_layers_model/2),
                              dtype="float32"
                              )
        self.performer = Llama(model_path=performer_model_path,
                               n_ctx=max_token_observed,
                               logits_all=True,
                               verbose=False,
                               n_gpu_layers=int(n_gpu_layers_model/2),
                               dtype="float32"
                               )

        self.pad_id = self.observer.token_eos() if hasattr(self.observer, 'token_eos') else 2 # 11 if falcon
        self.control_token_ids = get_control_token_ids(self.observer)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def change_mode(self, mode: str):
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _tokenize(self, texts: list[str]):
        ids_batch, mask_batch = [], []
        for txt in texts:
            ids = self.observer.tokenize(txt.encode("utf-8"), add_bos=True)
            ids = ids[:self.max_token_observed]
            ids_batch.append(ids)
            mask_batch.append([1] * len(ids))
        maxlen = self.max_token_observed
        pad_id = self.pad_id
        ids_padded = [seq + [pad_id]*(maxlen - len(seq)) for seq in ids_batch]
        mask_padded = [m + [0]*(maxlen - len(m)) for m in mask_batch]
        return np.array(ids_padded, dtype=np.int32), np.array(mask_padded, dtype=np.int32)

    def _get_logits(self, model: Llama, input_ids: np.ndarray):
        out = []
        for seq in input_ids:
            model.reset()
            model.eval(seq.tolist())
            logits = np.array(model.scores)
            if logits.dtype != np.float32:
                logits = logits.astype(np.float32)
            out.append(logits)
        return np.stack(out)

    def compute_encodings_score(self, enc_ids_mask: tuple[np.ndarray, np.ndarray]):
        input_ids, attention_mask = enc_ids_mask
        fut_o = self.executor.submit(self._get_logits, self.observer, input_ids)
        fut_p = self.executor.submit(self._get_logits, self.performer, input_ids)
        obs_logits = fut_o.result()
        perf_logits = fut_p.result()
        ppl = perplexity(input_ids, perf_logits, attention_mask, pad_token_id=self.pad_id, control_token_ids=self.control_token_ids)
        ent = entropy(obs_logits, perf_logits, input_ids, self.pad_id, attention_mask, control_token_ids=self.control_token_ids)
        return ppl / ent

    def compute_score(self, input_text: Union[str, list[str]]):
        texts = [input_text] if isinstance(input_text, str) else input_text
        enc = self._tokenize(texts)
        sc = self.compute_encodings_score(enc)
        return float(sc[0]) if isinstance(input_text, str) else sc.tolist()

    def predict(self, input_text: Union[str, list[str]]):
        scr = self.compute_score(input_text)
        if isinstance(scr, float):
            return "Human" if scr >= self.threshold else "AI"
        elif isinstance(scr, (list, np.ndarray)):
            return ["Human" if s >= self.threshold else "AI" for s in scr]
        else:
            return scr
