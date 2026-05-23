import torch
from config import MAX_LEN, device
from model.tokenizer import tok, SOS, EOS, PAD, SEP, USR, AST


def predict(model, history, text, max_new=128, temperature=1, top_k=50):
    with torch.no_grad():
        ctx_ids = [SOS]
        for msg in history:
            role_tok = USR if msg["role"] == "user" else AST
            ctx_ids += [role_tok] + tok.encode(msg["content"], add_special_tokens=False) + [SEP]
        ctx_ids += [USR] + tok.encode(text, add_special_tokens=False) + [SEP, AST]

        if len(ctx_ids) > MAX_LEN - max_new:
            ctx_ids = [SOS] + ctx_ids[-(MAX_LEN - max_new):]

        tokens = torch.tensor([ctx_ids], device=device)
        result = []
        for _ in range(max_new):
            if tokens.size(1) >= MAX_LEN:
                break
            logits              = model(tokens)
            logits              = logits[:, -1] / temperature
            top_logits, top_idx = torch.topk(logits, top_k)
            probs               = torch.softmax(top_logits, dim=-1)
            next_tok            = top_idx.gather(-1, torch.multinomial(probs, 1))
            if next_tok.item() in (PAD, EOS):
                break
            result.append(next_tok.item())
            tokens = torch.cat([tokens, next_tok], dim=1)

    return tok.decode(result, skip_special_tokens=True)
