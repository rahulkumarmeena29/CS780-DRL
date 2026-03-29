from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DuelingDDQN(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)
        adv = self.adv_stream(feat)
        return value + adv - adv.mean(dim=1, keepdim=True)

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition: Transition):
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]

        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        s = np.stack([t.s for t in samples]).astype(np.float32)
        a = np.array([t.a for t in samples], dtype=np.int64)
        r = np.array([t.r for t in samples], dtype=np.float32)
        s2 = np.stack([t.s2 for t in samples]).astype(np.float32)
        d = np.array([t.done for t in samples], dtype=np.float32)
        w = np.array(weights, dtype=np.float32)

        return s, a, r, s2, d, w, indices

    def update_priorities(self, indices, td_errors):
        td_errors = np.abs(td_errors) + self.eps
        for idx, prio in zip(indices, td_errors):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)
    
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--target_sync", type=int, default=250)
    ap.add_argument("--train_freq", type=int, default=4)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--per_alpha", type=float, default=0.6)
    ap.add_argument("--per_beta_start", type=float, default=0.4)
    ap.add_argument("--per_beta_end", type=float, default=1.0)
    ap.add_argument("--per_eps", type=float, default=1e-6)
    ap.add_argument("--reward_clip", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    OBELIX = import_obelix(args.obelix_py)
    q = DuelingDDQN().to(device)
    tgt = DuelingDDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplayBuffer(
        capacity=args.replay,
        alpha=args.per_alpha,
        eps=args.per_eps,
    )

    steps = 0

    def eps_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    def beta_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.per_beta_end
        frac = t / args.eps_decay_steps
        return args.per_beta_start + frac * (args.per_beta_end - args.per_beta_start)

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        s = env.reset(seed=args.seed + ep)
        s = np.asarray(s, dtype=np.float32)
        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(s).unsqueeze(0).to(device)
                    qvals = q(s_t)
                    a = int(torch.argmax(qvals, dim=1).item())

            s2, r, done = env.step(ACTIONS[a], render=False)
            s2 = np.asarray(s2, dtype=np.float32)
            r_clipped = float(np.clip(r, -args.reward_clip, args.reward_clip))
            replay.add(
                Transition(
                    s=s,
                    a=a,
                    r=r_clipped,
                    s2=s2,
                    done=bool(done),
                )
            )
            ep_ret += float(r)  # log raw reward, not clipped
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch) and steps % args.train_freq == 0:
                beta = beta_by_step(steps)
                sb, ab, rb, s2b, db, wb, indices = replay.sample(args.batch, beta=beta)
                sb_t = torch.from_numpy(sb).to(device)
                ab_t = torch.from_numpy(ab).to(device)
                rb_t = torch.from_numpy(rb).to(device)
                s2b_t = torch.from_numpy(s2b).to(device)
                db_t = torch.from_numpy(db).to(device)
                wb_t = torch.from_numpy(wb).to(device)
                with torch.no_grad():
                    next_actions = q(s2b_t).argmax(dim=1, keepdim=True)
                    next_q_target = tgt(s2b_t).gather(1, next_actions).squeeze(1)
                    targets = rb_t + args.gamma * (1.0 - db_t) * next_q_target
                current_q = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                td_errors = targets - current_q
                per_sample_loss = nn.functional.smooth_l1_loss(
                    current_q, targets, reduction="none"
                )
                loss = (wb_t * per_sample_loss).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                replay.update_priorities(
                    indices,
                    td_errors.detach().abs().cpu().numpy(),
                )
                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}/{args.episodes} "
                f"return={ep_ret:.2f} "
                f"eps={eps_by_step(steps):.3f} "
                f"beta={beta_by_step(steps):.3f} "
                f"replay={len(replay)} "
                f"steps={steps}"
            )
    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()