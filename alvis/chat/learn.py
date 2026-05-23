import torch
from config import MAX_LEN, GRAD_STEPS, STYLE_STEPS, device
from model.tokenizer import tok, SOS, EOS, PAD, SEP, USR, AST, VOCAB_SIZE, criterion


def make_sample(user_text, asst_text):
    u_ids = tok.encode(user_text, add_special_tokens=False)
    a_ids = tok.encode(asst_text, add_special_tokens=False)
    ctx   = [SOS, USR] + u_ids + [SEP, AST]
    seq   = (ctx + a_ids + [EOS])[:MAX_LEN + 1]
    x     = seq[:-1]
    y     = list(seq[1:])
    for j in range(min(len(ctx) - 1, len(y))):
        y[j] = PAD
    pad = MAX_LEN - len(x)
    x  += [PAD] * pad
    y  += [PAD] * pad
    return torch.tensor([x], device=device), torch.tensor([y], device=device)


def learn(model, optimizer, user_text, asst_text, steps=GRAD_STEPS):
    x, y = make_sample(user_text, asst_text)
    if (y != PAD).sum() == 0:
        return
    model.train()
    for _ in range(steps):
        logits = model(x)
        loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()


def learn_style(model, optimizer, user_text):
    ids = tok.encode(user_text, add_special_tokens=False)
    if len(ids) < 2:
        return
    seq = ([SOS] + ids + [EOS])[:MAX_LEN + 1]
    pad = MAX_LEN - (len(seq) - 1)
    x   = torch.tensor([seq[:-1] + [PAD] * pad], device=device)
    y   = torch.tensor([seq[1:]  + [PAD] * pad], device=device)
    model.train()
    for _ in range(STYLE_STEPS):
        logits = model(x)
        loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
