from __future__ import annotations
from typing import List
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18

# ======================= SAC ACTOR =======================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, N_ACTIONS),
        )

    def forward(self, x):
        probs = F.softmax(self.net(x), dim=-1)
        return probs, F.log_softmax(self.net(x), dim=-1)

# ======================= INFERENCE =======================
_model = None

def _load_once():
    global _model
    if _model is not None:
        return

    here  = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. "
            "Train offline with train_sac.py and include it in the submission zip."
        )

    sd = torch.load(wpath, map_location="cpu")
    
    # Unwrap checkpoint dict if saved that way
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    m = PolicyNetwork()
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    probs, _ = _model(x)
    probs = probs.squeeze(0).cpu().numpy()
    
    best = int(rng.choice(N_ACTIONS, p=probs / probs.sum()))
    
    return ACTIONS[best]
