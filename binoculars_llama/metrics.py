import torch
import numpy as np

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def perplexity(input_ids: np.ndarray, logits: np.ndarray, attention_mask: np.ndarray, pad_token_id=None, control_token_ids=None):
    logits = torch.tensor(logits, dtype=torch.float32)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    shifted_logits = logits[..., :-1, :].float()
    shifted_labels = input_ids[..., 1:]
    shifted_attention_mask = attention_mask[..., 1:]

    min_len = min(shifted_logits.shape[1], shifted_labels.shape[1], shifted_attention_mask.shape[1])
    shifted_logits = shifted_logits[:, :min_len, :].float()
    shifted_labels = shifted_labels[:, :min_len]
    shifted_attention_mask = shifted_attention_mask[:, :min_len]

    mask = torch.as_tensor(shifted_attention_mask, dtype=torch.long).detach().clone()
    if pad_token_id is not None:
        mask = mask * (shifted_labels != pad_token_id)
    if control_token_ids is not None and len(control_token_ids) > 0:
        for tid in control_token_ids:
            mask = mask * (shifted_labels != tid)
    ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * mask).sum(1) / mask.sum(1)
    ppl = ppl.to("cpu").float().numpy().astype(np.float32)
    return ppl


def entropy(p_logits, q_logits, input_ids, pad_token_id, attention_mask=None, control_token_ids=None):
    p_logits = torch.tensor(p_logits, dtype=torch.float32)
    q_logits = torch.tensor(q_logits, dtype=torch.float32)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    if attention_mask is None:
        mask = (input_ids != pad_token_id).type(torch.uint8)
    else:
        mask = torch.tensor(attention_mask, dtype=torch.long)

    min_len = min(p_logits.shape[1], q_logits.shape[1], input_ids.shape[1], mask.shape[1])
    p_logits = p_logits[:, :min_len, :].float()
    q_logits = q_logits[:, :min_len, :].float()
    input_ids = input_ids[:, :min_len]
    mask = mask[:, :min_len]

    if control_token_ids is not None and len(control_token_ids) > 0:
        for tid in control_token_ids:
            mask = mask * (input_ids != tid)
    vocab_size = p_logits.shape[-1]
    p_proba = softmax_fn(p_logits).view(-1, vocab_size).float()
    q_scores = q_logits.view(-1, vocab_size).float()
    ce = ce_loss_fn(input=q_scores, target=p_proba).view(input_ids.shape[0], -1).float()
    padding_mask = mask.float()
    agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy().astype(np.float32))
    return agg_ce