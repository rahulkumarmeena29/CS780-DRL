from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

_model: Optional[PolicyNet] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 2
_CLOSE_LOGIT_DELTA = 0.05

def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m = PolicyNet()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits = _model(x).squeeze(0).cpu().numpy()
    best = int(np.argmax(logits))
    if _last_action is not None:
        order = np.argsort(-logits)
        best_logit = float(logits[order[0]])
        second_logit = float(logits[order[1]])
        if (best_logit - second_logit) < _CLOSE_LOGIT_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]