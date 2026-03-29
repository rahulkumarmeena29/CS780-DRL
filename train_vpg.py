from __future__ import annotations
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

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

class ValueNet(nn.Module):
    def __init__(self, in_dim: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return np.array(returns, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy_lr", type=float, default=3e-4)
    ap.add_argument("--value_lr", type=float, default=7e-4)
    ap.add_argument("--entropy_coef", type=float, default=0.05)
    ap.add_argument("--normalize_returns", action="store_true", default=False)
    ap.add_argument("--normalize_advantages", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    OBELIX = import_obelix(args.obelix_py)
    policy_net = PolicyNet().to(device)
    value_net = ValueNet().to(device)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_lr)

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
        states = []
        log_probs = []
        rewards = []
        entropies = []
        ep_ret = 0.0

        for _ in range(args.max_steps):
            s_t = torch.from_numpy(s).float().unsqueeze(0).to(device)
            logits = policy_net(s_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            act_idx = action.item()

            s2, r, done = env.step(ACTIONS[act_idx], render=False)
            
            s2 = np.asarray(s2, dtype=np.float32)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            states.append(s.copy())
            rewards.append(float(r))
            ep_ret += float(r)
            s = s2
            if done:
                break

        if len(rewards) == 0:
            continue

        returns = compute_returns(rewards, args.gamma)
        returns_t = torch.from_numpy(returns).float().to(device)
        if args.normalize_returns and len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        states_t = torch.from_numpy(np.stack(states)).float().to(device)
        values_t = value_net(states_t)
        advantages_t = returns_t - values_t.detach()
        if args.normalize_advantages and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        log_probs_t = torch.cat(log_probs)
        entropies_t = torch.cat(entropies)
        policy_loss = -(log_probs_t * advantages_t).mean() - args.entropy_coef * entropies_t.mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
        policy_optimizer.step()
        value_loss = nn.functional.smooth_l1_loss(values_t, returns_t, beta=0.5)
        value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), 5.0)
        value_optimizer.step()

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}/{args.episodes} "
                f"return={ep_ret:.2f} "
                f"policy_loss={policy_loss.item():.4f} "
                f"value_loss={value_loss.item():.4f}"
            )

    torch.save(policy_net.state_dict(), args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()