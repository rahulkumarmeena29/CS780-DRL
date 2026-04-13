from __future__ import annotations
import argparse
import random
import os
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, n_actions), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def sensor_potential(obs: np.ndarray) -> float:
    phi  = 1.0 * float(obs[0])    # L-far-1
    phi += 1.0 * float(obs[1])    # L-far-2
    phi += 2.0 * float(obs[2])    # L-near-1
    phi += 2.0 * float(obs[3])    # L-near-2
    phi += 3.0 * float(obs[4])    # F-far-1
    phi += 5.0 * float(obs[5])    # F-near-1
    phi += 3.0 * float(obs[6])    # F-far-2
    phi += 5.0 * float(obs[7])    # F-near-2
    phi += 3.0 * float(obs[8])    # F-far-3
    phi += 5.0 * float(obs[9])    # F-near-3
    phi += 3.0 * float(obs[10])   # F-far-4
    phi += 5.0 * float(obs[11])   # F-near-4
    phi += 2.0 * float(obs[12])   # R-near-1
    phi += 2.0 * float(obs[13])   # R-near-2
    phi += 1.0 * float(obs[14])   # R-far-1
    phi += 1.0 * float(obs[15])   # R-far-2
    phi += 10.0 * float(obs[16])  # goal sensor
    phi -= 5.0 * float(obs[17])   # hazard sensor
    return phi

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plotResults(plot_history, all_ep_rets, args):
    if plot_history:
        ep_nums, means = zip(*plot_history)
        raw_eps, raw_rets = zip(*all_ep_rets) if all_ep_rets else ([], [])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(raw_eps, raw_rets, color="green", s=4, alpha=0.35, label="Episode return")
        ax.plot(ep_nums, means, color="red", linewidth=2.0, label="Rolling mean (50 eps)")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title(f"VPG Training — D{args.difficulty} {'+ walls' if args.wall_obstacles else ''}")
        ax.legend()
        plot_path = args.out.replace(".pth", "_curve.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved training curve → {plot_path}")

def save_checkpoint(path, policy_net, value_net):
    # Saves the ENTIRE agent state so curriculum training can resume flawlessly
    checkpoint = {
        "state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict()
    }
    torch.save(checkpoint, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--episodes_per_update", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy_lr", type=float, default=3e-4) 
    ap.add_argument("--value_lr", type=float, default=1e-3) 
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--normalize_returns", action="store_true")
    ap.add_argument("--normalize_advantages", action="store_true")
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--frame_stack", type=int, default=32)
    ap.add_argument("--reward_scale", type=float, default=0.005) 
    ap.add_argument("--reward_clip", type=float, default=10.0)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--value_epochs", type=int, default=10)
    ap.add_argument("--load_weights", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    OBELIX = import_obelix(args.obelix_py)
    in_dim = 18 * args.frame_stack
    
    policy_net = PolicyNet(in_dim=in_dim).to(device)
    value_net = ValueNet(in_dim=in_dim).to(device)
    
    with torch.no_grad():
        policy_net.net[-1].bias.data[2] += 0.5
        
    # Curriculum Learning: Load weights if provided
    if args.load_weights:
        checkpoint = torch.load(args.load_weights, map_location=device)
        if isinstance(checkpoint, dict) and "value_state_dict" in checkpoint:
            policy_net.load_state_dict(checkpoint["state_dict"])
            value_net.load_state_dict(checkpoint["value_state_dict"])
            print(f"Loaded FULL VPG checkpoint from {args.load_weights} (Curriculum mode active)")
        else:
            sd = checkpoint["state_dict"] if (isinstance(checkpoint, dict) and "state_dict" in checkpoint) else checkpoint
            policy_net.load_state_dict(sd, strict=True)
            print(f"Warning: Loaded ONLY actor weights from {args.load_weights}. Value network randomly initialized.")

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_lr)
    
    best_train_return = -float("inf")
    total_episodes_done = 0
    
    # Bookkeeping for plotting
    all_ep_rets = []
    plot_history = []
    last50_rets = []

    while total_episodes_done < args.episodes:
        batch_states = []
        batch_log_probs = []
        batch_entropies = []
        batch_returns = []
        batch_advantages = []
        train_returns_this_round = []
        episodes_this_round = min(args.episodes_per_update, args.episodes - total_episodes_done)

        for ep_local in range(episodes_this_round):
            ep = total_episodes_done + ep_local
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
            stack = deque(maxlen=args.frame_stack)
            for _ in range(args.frame_stack):
                stack.append(s.copy())

            states = []
            log_probs = []
            rewards = []
            entropies = []
            values = []
            dones = []
            ep_ret = 0.0

            for _ in range(args.max_steps):
                s_stacked = np.concatenate(stack).astype(np.float32)
                s_t = torch.from_numpy(s_stacked).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    value = value_net(s_t)
                
                logits = policy_net(s_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                act_idx = action.item()
                
                s2, r, done = env.step(ACTIONS[act_idx], render=False)
                s2 = np.asarray(s2, dtype=np.float32)
                ep_ret += float(r)
                
                phi_s = sensor_potential(s)
                phi_s2 = sensor_potential(s2)
                r_shaped = float(r) + args.gamma * phi_s2 - phi_s
                r_shaped *= args.reward_scale
                r_shaped = float(np.clip(r_shaped, -args.reward_clip, args.reward_clip))

                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                states.append(s_stacked.copy())
                rewards.append(r_shaped)
                values.append(float(value.item()))
                dones.append(float(done))
                
                s = s2
                stack.append(s2)

                if done:
                    break
            
            # Record keeping for this episode
            all_ep_rets.append((ep, ep_ret))
            last50_rets.append(ep_ret)
            if len(last50_rets) > 50:
                last50_rets.pop(0)

            if len(rewards) == 0:
                continue

            s_stacked = np.concatenate(stack).astype(np.float32)
            s_t = torch.from_numpy(s_stacked).unsqueeze(0).to(device)
            with torch.no_grad():
                last_value = float(value_net(s_t).item())

            values_for_gae = np.array(values + [last_value], dtype=np.float32)
            advantages, returns = compute_gae(
                rewards=np.array(rewards, dtype=np.float32),
                values=values_for_gae,
                dones=np.array(dones, dtype=np.float32),
                gamma=args.gamma,
                lam=args.gae_lambda
            )
            returns_t = torch.from_numpy(returns).float().to(device)
            advantages_t = torch.from_numpy(advantages).float().to(device)
            states_t = torch.from_numpy(np.stack(states)).float().to(device)
            
            batch_states.append(states_t)
            batch_log_probs.append(torch.cat(log_probs))
            batch_entropies.append(torch.cat(entropies))
            batch_returns.append(returns_t)
            batch_advantages.append(advantages_t)
            train_returns_this_round.append(ep_ret)
            
            if ep_ret > best_train_return:
                best_train_return = ep_ret
                save_checkpoint(args.out, policy_net, value_net)

        total_episodes_done += episodes_this_round
        
        if len(batch_states) == 0:
            continue
            
        states_t = torch.cat(batch_states, dim=0)
        log_probs_t = torch.cat(batch_log_probs, dim=0)
        entropies_t = torch.cat(batch_entropies, dim=0)
        returns_t = torch.cat(batch_returns, dim=0)
        advantages_t = torch.cat(batch_advantages, dim=0)
        
        if args.normalize_advantages and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
        policy_loss = -(log_probs_t * advantages_t.detach()).mean() - args.entropy_coef * entropies_t.mean()
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), args.max_grad_norm)
        policy_optimizer.step()

        value_loss_final = 0.0
        for _ in range(args.value_epochs):
            values_t = value_net(states_t)
            value_loss = nn.functional.mse_loss(values_t, returns_t)
            value_optimizer.zero_grad()
            (args.value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
            value_optimizer.step()
            value_loss_final = value_loss.item()

        # Generate output logs
        mean_train_return = float(np.mean(train_returns_this_round)) if train_returns_this_round else float("nan")
        mean_50 = np.mean(last50_rets) if last50_rets else mean_train_return
        
        plot_history.append((total_episodes_done, mean_50))

        print(
            f"Episodes {total_episodes_done}/{args.episodes} "
            f"mean_50={mean_50:.1f} "
            f"best_train_return={best_train_return:.1f} "
            f"policy_loss={policy_loss.item():.4f} "
            f"value_loss={value_loss_final:.4f} "
            f"entropy={entropies_t.mean().item():.4f}"
        )
        
    print(f"\nTraining Complete! Best return: {best_train_return:.1f}")
    
    # Save the absolute final network state at the end of training
    save_checkpoint(args.out, policy_net, value_net)
    print(f"Saved full VPG checkpoint → {args.out}")

    # Plot the results
    plotResults(plot_history, all_ep_rets, args)

if __name__ == "__main__":
    main()
