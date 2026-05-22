import sys
import os
import subprocess

try:
    import argparse
    import json
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset, DataLoader
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError as e:
    print(f"Librairie manquante : {e}")
    installer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install_deps.py")
    subprocess.run([sys.executable, installer])
    print("\nRelancement d'ALVIS...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

HERE            = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME      = "antoinelouis/belgpt2"
CHECKPOINT        = os.path.join(HERE, "checkpoint.pt")
BEST_CHECKPOINT   = os.path.join(HERE, "checkpoint_best.pt")
DATASET           = os.path.join(HERE, "dataset.json")
CONVERSATIONS     = os.path.join(HERE, "conversations.json")

# ── Curriculum learning ───────────────────────────────────────────────────────
STAGES = [
    (20, 128),
    (40, 256),
    (80, 512),
]
TOTAL_EPOCHS  = STAGES[-1][0]
MAX_LEN_MODEL = max(seq_len for _, seq_len in STAGES)
BATCH_SIZE    = 4

# ── Chat / apprentissage en ligne ─────────────────────────────────────────────
MAX_LEN          = 256
LR               = 2e-5
GRAD_STEPS       = 1
CORRECTION_STEPS = 1
STYLE_STEPS      = 1
REPLAY_STEPS     = 1    # poids plus faible pour les échanges passés
REPLAY_BATCH     = 16    # échanges passés rejoués aléatoirement par tour
SAVE_EVERY       = 10
MAX_CORRECTIONS  = 2

# ── Chauffe initiale ──────────────────────────────────────────────────────────
WARMUP_EPOCHS     = 4
WARMUP_MAX_LEN    = 128
WARMUP_BATCH_SIZE = 4


# ── Tokenizer ─────────────────────────────────────────────────────────────────

tok = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tok.add_special_tokens({
    "pad_token": "<|pad|>",
    "additional_special_tokens": ["<|sep|>"],
})
SOS        = tok.bos_token_id
EOS        = tok.eos_token_id
PAD        = tok.pad_token_id
SEP        = tok.convert_tokens_to_ids("<|sep|>")
VOCAB_SIZE = len(tok)

criterion = nn.CrossEntropyLoss(ignore_index=PAD)


# ── Modèle ────────────────────────────────────────────────────────────────────

class ChatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        self.model.resize_token_embeddings(VOCAB_SIZE)

    def forward(self, x):
        return self.model(input_ids=x, attention_mask=(x != PAD).long()).logits


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChatDataset(Dataset):
    def __init__(self, data, max_len):
        self.samples = []
        greeting_cap   = 200
        greeting_count = 0
        greeting = "bonjour ! comment puis-je vous aider aujourd'hui ?"

        for item in data:
            if "messages" in item:
                msgs   = item["messages"]
                prefix = [SOS]
                i      = 0
                while i < len(msgs) - 1:
                    if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                        if msgs[i + 1]["content"].lower() == greeting:
                            if greeting_count >= greeting_cap:
                                i += 2; continue
                            greeting_count += 1

                        u_ids = tok.encode(msgs[i]["content"],     add_special_tokens=False)
                        a_ids = tok.encode(msgs[i + 1]["content"], add_special_tokens=False)

                        ctx = prefix + u_ids + [SEP]
                        seq = (ctx + a_ids + [EOS])[:max_len + 1]
                        if len(seq) < 4:
                            i += 2; continue

                        x = seq[:-1]
                        y = list(seq[1:])
                        for j in range(min(len(ctx) - 1, len(y))):
                            y[j] = PAD
                        if all(v == PAD for v in y):
                            i += 2; continue

                        pad = max_len - len(x)
                        x  += [PAD] * pad
                        y  += [PAD] * pad
                        self.samples.append((torch.tensor(x), torch.tensor(y)))

                        new_prefix = prefix + u_ids + [SEP] + a_ids + [SEP]
                        if len(new_prefix) > max_len // 2:
                            new_prefix = [SOS] + new_prefix[-(max_len // 2):]
                        prefix = new_prefix
                        i += 2
                    else:
                        i += 1

            elif "text" in item:
                ids = tok.encode(item["text"], add_special_tokens=False)
                if len(ids) >= 4:
                    seq = ([SOS] + ids)[:max_len + 1]
                    x   = (seq[:-1] + [PAD] * max_len)[:max_len]
                    y   = (seq[1:]  + [PAD] * max_len)[:max_len]
                    self.samples.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def build_loaders(seq_len, train_raw, val_raw):
    tr = ChatDataset(train_raw, max_len=seq_len)
    va = ChatDataset(val_raw,   max_len=seq_len)
    kw = dict(num_workers=0, pin_memory=device.type == "cuda")
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, **kw))


# ── Apprentissage en ligne ────────────────────────────────────────────────────

def make_sample(user_text, asst_text):
    u_ids = tok.encode(user_text, add_special_tokens=False)
    a_ids = tok.encode(asst_text, add_special_tokens=False)
    ctx   = [SOS] + u_ids + [SEP]
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


# ── Experience replay ─────────────────────────────────────────────────────────

def load_conversations():
    if os.path.exists(CONVERSATIONS):
        with open(CONVERSATIONS, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversation(past, user_text, asst_text, score=1.0):
    past.append({"user": user_text, "assistant": asst_text, "score": score})
    with open(CONVERSATIONS, "w", encoding="utf-8") as f:
        json.dump(past, f, ensure_ascii=False, indent=2)


def replay(model, optimizer, past):
    if len(past) < 2:
        return
    weights = [entry.get("score", 1.0) for entry in past]
    batch   = random.choices(past, weights=weights, k=min(REPLAY_BATCH, len(past)))
    seen    = set()
    for entry in batch:
        key = entry["user"] + "\x00" + entry["assistant"]
        if key in seen:
            continue
        seen.add(key)
        steps = CORRECTION_STEPS if entry.get("score", 1.0) > 1.0 else REPLAY_STEPS
        learn(model, optimizer, entry["user"], entry["assistant"], steps=steps)


# ── Génération ────────────────────────────────────────────────────────────────

def predict(model, history, text, max_new=128, temperature=1, top_k=50):
    with torch.no_grad():
        ctx_ids = [SOS]
        for msg in history:
            ctx_ids += tok.encode(msg["content"], add_special_tokens=False) + [SEP]
        ctx_ids += tok.encode(text, add_special_tokens=False) + [SEP]

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



# ── Eval (mode entraînement) ──────────────────────────────────────────────────

def evaluate(model, val_loader):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            mask   = y != PAD
            preds  = torch.argmax(logits, dim=-1)
            total_loss += loss.item()
            total_acc  += (preds[mask] == y[mask]).float().mean().item()
            n += 1
    model.train()
    return total_loss / n, total_acc / n


def save_curves(h):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(len(h["loss"]))
    ax1.plot(ep, h["loss"], label="Train")
    if h["val_loss"]: ax1.plot(ep, h["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(ep, h["acc"], label="Train")
    if h["val_acc"]:  ax2.plot(ep, h["val_acc"],  label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("Courbes sauvegardées dans training_curves.png")


# ── Helpers de stage ──────────────────────────────────────────────────────────

def get_seq_len(epoch):
    for epoch_end, seq_len in STAGES:
        if epoch < epoch_end:
            return seq_len
    return STAGES[-1][1]


def next_stage_start(current_epoch):
    current_end = None
    for epoch_end, _ in STAGES:
        if current_epoch < epoch_end:
            current_end = epoch_end
            break
    if current_end is None:
        return None
    for epoch_end, _ in STAGES:
        if epoch_end > current_end:
            return current_end
    return None


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_train(model, optimizer, scaler, epoch, h, best_val_loss, best_epoch):
    torch.save({
        "mode": "train", "epoch": epoch,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "history_loss": h["loss"], "history_acc": h["acc"],
        "history_val_loss": h["val_loss"], "history_val_acc": h["val_acc"],
        "best_val_loss": best_val_loss, "best_epoch": best_epoch,
    }, CHECKPOINT)


def save_best(model, epoch, best_val_loss):
    torch.save({"epoch": epoch, "model": model.state_dict()}, BEST_CHECKPOINT)
    print(f"  ★ Meilleur modèle sauvegardé (epoch {epoch}, val loss {best_val_loss:.4f})")


def save_chat(model, optimizer, exchanges):
    torch.save({
        "mode": "chat", "exchanges": exchanges,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    }, CHECKPOINT)
    print(f"  [Sauvegardé — {exchanges} échanges]")


# ── Chauffe ───────────────────────────────────────────────────────────────────

def warmup(model, optimizer):
    if not os.path.exists(DATASET):
        print("Dataset introuvable — chauffe ignorée.")
        return

    print(f"Chauffe sur le dataset ({WARMUP_EPOCHS} epochs, MAX_LEN={WARMUP_MAX_LEN})...")
    with open(DATASET, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = ChatDataset(raw, max_len=WARMUP_MAX_LEN)
    loader  = DataLoader(dataset, batch_size=WARMUP_BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  {len(dataset)} échantillons")

    total_batches = len(loader)
    model.train()
    for epoch in range(WARMUP_EPOCHS):
        total, n = 0, 0
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
            pct = n / total_batches
            bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
            print(f"\r  Epoch {epoch + 1}/{WARMUP_EPOCHS} [{bar}] {n}/{total_batches} | Loss {total / n:.4f}", end="", flush=True)
        print()
    model.eval()
    print("Chauffe terminée.\n")


# ── Mode entraînement ─────────────────────────────────────────────────────────

def train_mode():
    import msvcrt

    if not os.path.exists(DATASET):
        print(f"Dataset introuvable : {DATASET}")
        return

    with open(DATASET, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    val_raw   = raw_data[4::5]
    train_raw = [item for i, item in enumerate(raw_data) if i % 5 != 4]

    model     = ChatModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    h         = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    start_epoch   = 0
    best_val_loss = float("inf")
    best_epoch    = 0

    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        h["loss"]     = ckpt.get("history_loss",     [])
        h["acc"]      = ckpt.get("history_acc",      [])
        h["val_loss"] = ckpt.get("history_val_loss", [])
        h["val_acc"]  = ckpt.get("history_val_acc",  [])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch    = ckpt.get("best_epoch", 0)

        if get_seq_len(best_epoch) != get_seq_len(min(start_epoch, TOTAL_EPOCHS - 1)):
            best_val_loss = float("inf")
            print("Nouveau stage — meilleur val loss réinitialisé")

        if start_epoch >= TOTAL_EPOCHS:
            print(f"Entraînement terminé (epoch {start_epoch - 1}). Ajoutez un stage dans STAGES.")
            return
        elif h["val_loss"] and h["val_loss"][-1] > best_val_loss:
            if os.path.exists(BEST_CHECKPOINT):
                best_ckpt = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
                model.load_state_dict(best_ckpt["model"])
                print(f"Overfitting détecté — meilleur modèle chargé (epoch {best_epoch})")
            nxt = next_stage_start(start_epoch)
            if nxt is not None:
                start_epoch = nxt
                print(f"Passage au stage suivant → MAX_LEN={get_seq_len(nxt)} (epoch {nxt})")
        else:
            print(f"Reprise depuis l'epoch {start_epoch}")

    model.train()
    try:
        current_seq_len = 0
        loader = val_loader = None

        for epoch in range(start_epoch, TOTAL_EPOCHS):
            seq_len = get_seq_len(epoch)
            if seq_len != current_seq_len:
                if current_seq_len != 0 and os.path.exists(BEST_CHECKPOINT):
                    best_ckpt = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
                    model.load_state_dict(best_ckpt["model"])
                    best_val_loss = float("inf")
                    print("  → Meilleur modèle chargé pour le stage suivant")
                print(f"  → Stage MAX_LEN {current_seq_len} → {seq_len}")
                current_seq_len = seq_len
                loader, val_loader = build_loaders(seq_len, train_raw, val_raw)

            total_loss, total_acc, n_batches = 0, 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(x)
                    loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                mask  = y != PAD
                preds = torch.argmax(logits, dim=-1)
                total_loss += loss.item()
                total_acc  += (preds[mask] == y[mask]).float().mean().item()
                n_batches  += 1

            avg_loss = total_loss / n_batches
            avg_acc  = total_acc  / n_batches
            val_loss, val_acc = evaluate(model, val_loader)

            h["loss"].append(avg_loss)
            h["acc"].append(avg_acc)
            h["val_loss"].append(val_loss)
            h["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
                save_best(model, epoch, best_val_loss)

            print(f"Epoch {epoch} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

            if msvcrt.kbhit():
                key = msvcrt.getch().lower()
                if key == b'n':
                    nxt = next_stage_start(epoch)
                    if nxt is not None:
                        print(f"  [N] Passage forcé au stage suivant (epoch {nxt})")
                        save_train(model, optimizer, scaler, epoch, h, best_val_loss, best_epoch)
                        if os.path.exists(BEST_CHECKPOINT):
                            best_ckpt = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
                            model.load_state_dict(best_ckpt["model"])
                        best_val_loss   = float("inf")
                        current_seq_len = 0
                    else:
                        print("  [N] Déjà au dernier stage.")

            if epoch % 5 == 0:
                save_train(model, optimizer, scaler, epoch, h, best_val_loss, best_epoch)

    except KeyboardInterrupt:
        print(f"\nInterrompu à l'epoch {epoch}. Sauvegarde...")
        save_train(model, optimizer, scaler, epoch, h, best_val_loss, best_epoch)
        save_curves(h)
        raise SystemExit(0)

    save_train(model, optimizer, scaler, epoch, h, best_val_loss, best_epoch)
    print(f"Entraînement terminé à l'epoch {epoch}.")
    save_curves(h)


# ── Mode chat ─────────────────────────────────────────────────────────────────

def chat_mode():
    model     = ChatModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    exchanges = 0

    if os.path.exists(CHECKPOINT):
        try:
            ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            if ckpt.get("mode") == "chat" and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            exchanges = ckpt.get("exchanges", 0)
            print(f"Checkpoint chargé — {exchanges} échanges mémorisés")
        except Exception as e:
            print(f"Checkpoint corrompu ({e}), suppression et rechauffe...")
            os.remove(CHECKPOINT)
            warmup(model, optimizer)
            save_chat(model, optimizer, exchanges)
    else:
        warmup(model, optimizer)
        save_chat(model, optimizer, exchanges)

    model.eval()
    history = []
    past = load_conversations()
    print(f"  {len(past)} échange(s) passé(s) chargé(s) pour le replay")

    print("─" * 40)
    print("Chat avec ALVIS  |  'quit' pour quitter  |  'reset' pour effacer l'historique")
    print("─" * 40)

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSauvegarde...")
            save_chat(model, optimizer, exchanges)
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            save_chat(model, optimizer, exchanges)
            break
        if user_input.lower() == "reset":
            history.clear()
            print("  [Historique effacé]")
            continue

        response = predict(model, history, user_input)
        print(f"ALVIS : {response}\n")

        # Corrections optionnelles — jusqu'à MAX_CORRECTIONS variantes
        corrections = []
        print(f"  ↳ Corrections (une par ligne, Entrée vide pour terminer) :")
        for _ in range(MAX_CORRECTIONS):
            try:
                line = input(f"  [{len(corrections) + 1}] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            corrections.append(line)

        if corrections:
            for c in corrections:
                learn(model, optimizer, user_input, c, steps=CORRECTION_STEPS)
            actual_response = corrections[0]
            print(f"  [{len(corrections)} correction(s) mémorisée(s)]\n")
            score = 5.0
        else:
            learn(model, optimizer, user_input, response)
            actual_response = response
            score = 1.0

        learn_style(model, optimizer, user_input)
        save_conversation(past, user_input, actual_response, score)
        replay(model, optimizer, past)
        exchanges += 1

        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": actual_response})

        if exchanges % SAVE_EVERY == 0:
            save_chat(model, optimizer, exchanges)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALVIS — chatbot belgpt2")
    parser.add_argument("--train", action="store_true", help="Curriculum learning sur le dataset")
    args = parser.parse_args()

    if args.train:
        train_mode()
    else:
        chat_mode()