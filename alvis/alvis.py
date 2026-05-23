import sys
import os
import subprocess

# ── Vérification des dépendances pip ─────────────────────────────────────────
try:
    for _pkg in ["torch", "transformers", "matplotlib", "numpy"]:
        __import__(_pkg)
except ImportError as e:
    print(f"Dépendance manquante : {e}")
    print("Installation en cours...")
    for pkg in ["numpy", "torch", "torchvision", "matplotlib", "transformers"]:
        r = subprocess.run([sys.executable, "-m", "pip", "install", pkg],
                           capture_output=True, text=True)
        print(f"  [{'OK' if r.returncode == 0 else 'ERREUR'}] {pkg}")
    print("\nRelancement...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse
import torch
import torch.optim as optim

from config import CHECKPOINT, LR, SAVE_EVERY, MAX_CORRECTIONS, CORRECTION_STEPS, device
from model.architecture import ChatModel
from chat.learn import learn, learn_style
from chat.replay import load_conversations, save_conversation, replay
from chat.generate import predict
from chat.display import display_chat
from training.loops import train_mode, warmup
from training.checkpoint import save_chat


def _init_model():
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
    return model, optimizer, exchanges


def _get_corrections():
    corrections = []
    print("  ↳ Corrections (une par ligne, Entrée vide pour terminer) :")
    for _ in range(MAX_CORRECTIONS):
        try:
            line = input(f"  [{len(corrections) + 1}] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        corrections.append(line)
    return corrections


def _apply_turn(model, optimizer, past, user_input, response):
    corrections = _get_corrections()
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
    return actual_response


# ── Mode chat ─────────────────────────────────────────────────────────────────

def chat_mode():
    model, optimizer, exchanges = _init_model()
    history = []
    past    = load_conversations()
    print(f"  {len(past)} échange(s) passé(s) chargé(s) pour le replay")

    while True:
        display_chat(history)
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSauvegarde...")
            save_chat(model, optimizer, exchanges)
            break

        if not user_input:                          continue
        if user_input.lower() == "quit":
            save_chat(model, optimizer, exchanges); break
        if user_input.lower() == "reset":
            history.clear();                        continue
        if user_input.lower() == "/help":
            print("\n  Commandes :")
            print("  /help  — afficher cette aide")
            print("  reset  — effacer l'historique affiché (ALVIS garde sa mémoire)")
            print("  quit   — quitter et sauvegarder")
            print("\n  Après chaque réponse d'ALVIS, tapez une correction")
            print("  ou laissez vide pour valider la réponse telle quelle.")
            input("\n  [Entrée pour continuer] ");  continue

        response       = predict(model, history, user_input)
        display_chat(history, user_input, response)
        actual_response = _apply_turn(model, optimizer, past, user_input, response)
        exchanges      += 1
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
