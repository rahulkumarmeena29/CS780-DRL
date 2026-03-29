from __future__ import annotations
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy

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
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

def minibatch_indices(n, batch_size):
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        yield indices[start:start + batch_size]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sensor_potential(obs: np.ndarray) -> float:
    phi  = 1.0 * float(obs[0])   # L-far-1
    phi += 1.0 * float(obs[1])   # L-far-2
    phi += 2.0 * float(obs[2])   # L-near-1
    phi += 2.0 * float(obs[3])   # L-near-2
    phi += 3.0 * float(obs[4])   # F-far-1
    phi += 5.0 * float(obs[5])   # F-near-1
    phi += 3.0 * float(obs[6])   # F-far-2
    phi += 5.0 * float(obs[7])   # F-near-2
    phi += 3.0 * float(obs[8])   # F-far-3
    phi += 5.0 * float(obs[9])   # F-near-3
    phi += 3.0 * float(obs[10])  # F-far-4
    phi += 5.0 * float(obs[11])  # F-near-4
    phi += 2.0 * float(obs[12])  # R-near-1
    phi += 2.0 * float(obs[13])  # R-near-2
    phi += 1.0 * float(obs[14])  # R-far-1
    phi += 1.0 * float(obs[15])  # R-far-2
    phi += 10.0 * float(obs[16])
    phi -=  5.0 * float(obs[17])
    return phi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--episodes_per_update", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--policy_lr", type=float, default=1e-4)
    ap.add_argument("--value_lr", type=float, default=5e-4)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--ppo_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--reward_scale", type=float, default=0.05)
    ap.add_argument("--frame_stack", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best_out", type=str, default="best_policy.pth",
                    help="Path to save the richest best-policy checkpoint (default: best_policy.pth)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    OBELIX = import_obelix(args.obelix_py)
    
    in_dim_stacked = 18 * args.frame_stack
    policy_net = PolicyNet(in_dim=in_dim_stacked).to(device)
    
    # Reduced initial bias to 1.0 to prevent getting permanently stuck on walls
    policy_net.net[-1].bias.data[2] += 1.0

    value_net = ValueNet(in_dim=in_dim_stacked).to(device)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=args.policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=args.value_lr)
    best_raw_return = -float("inf")
    best_state_dict = None
    best_episode_num = 0
    total_episodes_done = 0
    cumulative_return = 0.0
    
    while total_episodes_done < args.episodes:
        buffer_states = []
        buffer_actions = []
        buffer_old_log_probs = []
        buffer_returns = []
        buffer_advantages = []
        batch_episode_returns = []
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
            
            # Initialize structure for Frame Stacking
            stack = deque(maxlen=args.frame_stack)
            for _ in range(args.frame_stack):
                stack.append(s)

            states = []
            actions = []
            log_probs_old = []
            rewards = []
            dones = []
            values = []
            ep_ret_raw = 0.0
            
            for _ in range(args.max_steps):
                s_stacked = np.concatenate(stack)
                s_t = torch.from_numpy(s_stacked).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = policy_net(s_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = value_net(s_t)

                act_idx = action.item()
                s2, r, done = env.step(ACTIONS[act_idx], render=False)
                s2 = np.asarray(s2, dtype=np.float32)
                ep_ret_raw += float(r)  # track raw return unmodified
                
                # Potential-based reward shaping requires the raw 18-dim sensor
                phi_s  = sensor_potential(s)
                phi_s2 = sensor_potential(s2)
                r_shaped = (float(r) + args.gamma * phi_s2 - phi_s) * args.reward_scale
                
                states.append(s_stacked.copy())
                actions.append(act_idx)
                log_probs_old.append(log_prob.item())
                rewards.append(r_shaped)
                dones.append(float(done))
                values.append(value.item())
                
                # Advance tracking
                s = s2
                stack.append(s2)
                if done:
                    break

            s_stacked = np.concatenate(stack)
            s_t = torch.from_numpy(s_stacked).float().unsqueeze(0).to(device)
            with torch.no_grad():
                last_value = value_net(s_t).item()
            values.append(last_value)
            
            if len(rewards) == 0:
                continue

            advantages, returns = compute_gae(
                rewards=np.array(rewards, dtype=np.float32),
                values=np.array(values, dtype=np.float32),
                dones=np.array(dones, dtype=np.float32),
                gamma=args.gamma,
                lam=args.gae_lambda,
            )
            buffer_states.extend(states)
            buffer_actions.extend(actions)
            buffer_old_log_probs.extend(log_probs_old)
            buffer_returns.extend(returns.tolist())
            buffer_advantages.extend(advantages.tolist())
            batch_episode_returns.append(ep_ret_raw)
            cumulative_return += ep_ret_raw
            if ep_ret_raw > best_raw_return:
                best_raw_return = ep_ret_raw
                best_state_dict = copy.deepcopy(policy_net.state_dict())
                best_episode_num = ep
        prev_total = total_episodes_done
        total_episodes_done += episodes_this_round

        if len(buffer_states) == 0:
            continue

        states_t = torch.tensor(np.array(buffer_states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(buffer_actions, dtype=torch.long, device=device)
        old_log_probs_t = torch.tensor(buffer_old_log_probs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(buffer_returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(buffer_advantages, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        n_samples = len(buffer_states)
        last_policy_loss = None
        last_value_loss = None

        for _ in range(args.ppo_epochs):
            for idx in minibatch_indices(n_samples, args.batch_size):
                batch_states = states_t[idx]
                batch_actions = actions_t[idx]
                batch_old_log_probs = old_log_probs_t[idx]
                batch_returns = returns_t[idx]
                batch_advantages = advantages_t[idx]
                logits = policy_net(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - args.entropy_coef * entropy
                values_pred = value_net(batch_states)
                value_loss = nn.functional.mse_loss(values_pred, batch_returns)
                policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), args.max_grad_norm)
                policy_optimizer.step()
                value_optimizer.zero_grad()
                (args.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
                value_optimizer.step()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()

        if prev_total // 50 < total_episodes_done // 50 or total_episodes_done == args.episodes:
            mean_ret = float(np.mean(batch_episode_returns)) if batch_episode_returns else float("nan")
            max_ret = float(np.max(batch_episode_returns)) if batch_episode_returns else float("nan")
            min_ret = float(np.min(batch_episode_returns)) if batch_episode_returns else float("nan")

            print(
                f"Episodes {total_episodes_done - episodes_this_round + 1}-{total_episodes_done}/{args.episodes} "
                f"mean_return={mean_ret:.2f} "
                f"max_return={max_ret:.2f} "
                f"min_return={min_ret:.2f} "
                f"policy_loss={last_policy_loss:.4f} "
                f"value_loss={last_value_loss:.4f} "
                f"best_return={best_raw_return:.2f}"
            )

    print(f"\nTraining Complete!")
    print(f"Cumulative Return over {total_episodes_done} episodes: {cumulative_return:.2f}")
    
    # Save the final fully-trained model to weights.pth
    torch.save(policy_net.state_dict(), args.out)
    print("Saved final trained policy to:", args.out)
    
    if best_state_dict is not None:
        torch.save({
            "episode": best_episode_num,
            "state_dict": best_state_dict,
            "best_raw_return": best_raw_return,
        }, args.best_out)
        print("Saved anomalous 'best single episode' checkpoint to:", args.best_out)

if __name__ == "__main__":
    main()