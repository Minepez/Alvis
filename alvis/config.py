import os
import torch

HERE            = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME      = "antoinelouis/belgpt2"
CHECKPOINT      = os.path.join(HERE, "checkpoint.pt")
BEST_CHECKPOINT = os.path.join(HERE, "checkpoint_best.pt")
DATASET         = os.path.join(HERE, "dataset.json")
CONVERSATIONS   = os.path.join(HERE, "conversations.json")

# ── Curriculum learning ───────────────────────────────────────────────────────
STAGES            = [(20, 128), (40, 256), (80, 512)]
TOTAL_EPOCHS      = STAGES[-1][0]
MAX_LEN_MODEL     = max(s for _, s in STAGES)
BATCH_SIZE        = 4

# ── Chat / apprentissage en ligne ─────────────────────────────────────────────
MAX_LEN           = 256
LR                = 2e-5
GRAD_STEPS        = 1
CORRECTION_STEPS  = 1
STYLE_STEPS       = 1
REPLAY_STEPS      = 1
REPLAY_BATCH      = 64
SAVE_EVERY        = 10
MAX_CORRECTIONS   = 2
DISPLAY_TURNS     = 3

# ── Chauffe initiale ──────────────────────────────────────────────────────────
WARMUP_EPOCHS     = 2
WARMUP_MAX_LEN    = 128
WARMUP_BATCH_SIZE = 4

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
