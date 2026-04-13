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
    phi  = 1.0 * float(obs[0])    + 1.0 * float(obs[1])
    phi += 2.0 * float(obs[2])    + 2.0 * float(obs[3])
    phi += 3.0 * float(obs[4])    + 5.0 * float(obs[5])
    phi += 3.0 * float(obs[6])    + 5.0 * float(obs[7])
    phi += 3.0 * float(obs[8])    + 5.0 * float(obs[9])
    phi += 3.0 * float(obs[10])   + 5.0 * float(obs[11])
    phi += 2.0 * float(obs[12])   + 2.0 * float(obs[13])
    phi += 1.0 * float(obs[14])   + 1.0 * float(obs[15])
    phi += 10.0 * float(obs[16])  - 5.0 * float(obs[17])
    return phi

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

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# =====================================================================
# VPG AGENT
# =====================================================================
class VPG:
    def __init__(self, obelix_class, env_kwargs, args, device="cpu"):
        self.obelix_class = obelix_class
        self.env_kwargs = env_kwargs
        self.args = args
        self.device = device
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.MAX_TRAIN_EPISODES = args.episodes
        self.MAX_EVAL_EPISODES = 10
        
        in_dim = 18 * args.frame_stack
        self.policyNetwork = PolicyNet(in_dim=in_dim).to(device)
        self.valueNetwork = ValueNet(in_dim=in_dim).to(device)
        
        with torch.no_grad():
            self.policyNetwork.net[-1].bias.data[2] += 0.5
            
        if args.load_weights:
            checkpoint = torch.load(args.load_weights, map_location=device)
            if isinstance(checkpoint, dict) and "value_state_dict" in checkpoint:
                self.policyNetwork.load_state_dict(checkpoint["state_dict"])
                self.valueNetwork.load_state_dict(checkpoint["value_state_dict"])
                print(f"Loaded FULL VPG checkpoint from {args.load_weights}")
            else:
                sd = checkpoint["state_dict"] if (isinstance(checkpoint, dict) and "state_dict" in checkpoint) else checkpoint
                self.policyNetwork.load_state_dict(sd, strict=True)
                print(f"Loaded actor weights from {args.load_weights}")

        self.policyOptimizer = optim.Adam(self.policyNetwork.parameters(), lr=args.policy_lr)
        self.valueOptimizer = optim.Adam(self.valueNetwork.parameters(), lr=args.value_lr)
        
        self.initBookKeeping()

    def initBookKeeping(self):
        self.n_done = 0
        self.best_return = -float("inf")
        self.plot_history = []
        self.all_ep_rets = []
        self.last50_ret = []

    def performBookKeeping(self, train=True, ep_ret=0):
        if train:
            self.last50_ret.append(ep_ret)
            if len(self.last50_ret) > 50:
                self.last50_ret.pop(0)
            
            self.all_ep_rets.append((self.n_done, ep_ret))
            self.n_done += 1
            
            if ep_ret > self.best_return:
                self.best_return = ep_ret
                checkpoint = {
                    "state_dict": self.policyNetwork.state_dict(),
                    "value_state_dict": self.valueNetwork.state_dict()
                }
                torch.save(checkpoint, self.args.out)
        else:
            self.final_eval_score = ep_ret

    def runVPG(self):
        plot_history, all_ep_rets = self.trainAgent()
        mean_eval, std_eval = self.evaluateAgent()
        return plot_history, all_ep_rets, mean_eval, std_eval

    def trainAgent(self):
        while self.n_done < self.MAX_TRAIN_EPISODES:
            batch_states, batch_log_probs, batch_entropies = [], [], []
            batch_returns, batch_advantages = [], []
            episodes_this_round = min(self.args.episodes_per_update, self.MAX_TRAIN_EPISODES - self.n_done)

            for ep_local in range(episodes_this_round):
                ep = self.n_done
                env = self.obelix_class(**self.env_kwargs, seed=self.args.seed + ep)
                s = env.reset(seed=self.args.seed + ep)
                s = np.asarray(s, dtype=np.float32)
                
                stack = deque(maxlen=self.args.frame_stack)
                for _ in range(self.args.frame_stack): stack.append(s.copy())
                states, log_probs, rewards, entropies, values, dones = [], [], [], [], [], []
                ep_ret = 0.0
                for _ in range(self.args.max_steps):
                    s_stacked = np.concatenate(stack).astype(np.float32)
                    s_t = torch.from_numpy(s_stacked).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        value = self.valueNetwork(s_t)
                    
                    logits = self.policyNetwork(s_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    act_idx = action.item()
                    
                    s2, r, done = env.step(ACTIONS[act_idx], render=False)
                    s2 = np.asarray(s2, dtype=np.float32)
                    ep_ret += float(r)
                    
                    phi_s = sensor_potential(s)
                    phi_s2 = sensor_potential(s2)
                    r_shaped = float(r) + self.gamma * phi_s2 - phi_s
                    r_shaped *= self.args.reward_scale
                    r_shaped = float(np.clip(r_shaped, -self.args.reward_clip, self.args.reward_clip))

                    log_probs.append(dist.log_prob(action))
                    entropies.append(dist.entropy())
                    states.append(s_stacked.copy())
                    rewards.append(r_shaped)
                    values.append(float(value.item()))
                    dones.append(float(done))
                    
                    s = s2
                    stack.append(s2)

                    if done: break
                
                self.performBookKeeping(train=True, ep_ret=ep_ret)

                if len(rewards) == 0: continue

                s_stacked = np.concatenate(stack).astype(np.float32)
                s_t = torch.from_numpy(s_stacked).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    last_value = float(self.valueNetwork(s_t).item())

                values_for_gae = np.array(values + [last_value], dtype=np.float32)
                advantages, returns = compute_gae(
                    rewards=np.array(rewards, dtype=np.float32),
                    values=values_for_gae,
                    dones=np.array(dones, dtype=np.float32),
                    gamma=self.gamma,
                    lam=self.gae_lambda
                )
                
                batch_states.append(torch.from_numpy(np.stack(states)).float().to(self.device))
                batch_log_probs.append(torch.cat(log_probs))
                batch_entropies.append(torch.cat(entropies))
                batch_returns.append(torch.from_numpy(returns).float().to(self.device))
                batch_advantages.append(torch.from_numpy(advantages).float().to(self.device))
            
            if len(batch_states) > 0:
                states_t = torch.cat(batch_states, dim=0)
                log_probs_t = torch.cat(batch_log_probs, dim=0)
                entropies_t = torch.cat(batch_entropies, dim=0)
                returns_t = torch.cat(batch_returns, dim=0)
                advantages_t = torch.cat(batch_advantages, dim=0)
                
                p_loss, v_loss, ent = self.trainNetworks(states_t, log_probs_t, entropies_t, returns_t, advantages_t)
                
                mean_50 = np.mean(self.last50_ret) if self.last50_ret else 0.0
                self.plot_history.append((self.n_done, mean_50))
                
                print(
                    f"Episodes {self.n_done}/{self.MAX_TRAIN_EPISODES} "
                    f"mean_50={mean_50:.1f} "
                    f"best_train_return={self.best_return:.1f} "
                    f"policy_loss={p_loss:.4f} "
                    f"value_loss={v_loss:.4f} "
                    f"entropy={ent:.4f}"
                )
                
        return self.plot_history, self.all_ep_rets

    def trainNetworks(self, states_t, log_probs_t, entropies_t, returns_t, advantages_t):
        if self.args.normalize_advantages and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
        policy_loss = -(log_probs_t * advantages_t.detach()).mean() - self.args.entropy_coef * entropies_t.mean()
        
        self.policyOptimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.args.max_grad_norm)
        self.policyOptimizer.step()

        value_loss_final = 0.0
        for _ in range(self.args.value_epochs):
            values_t = self.valueNetwork(states_t)
            value_loss = nn.functional.mse_loss(values_t, returns_t)
            self.valueOptimizer.zero_grad()
            (self.args.value_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(self.valueNetwork.parameters(), self.args.max_grad_norm)
            self.valueOptimizer.step()
            value_loss_final = value_loss.item()
            
        return policy_loss.item(), value_loss_final, entropies_t.mean().item()

    def evaluateAgent(self):
        print("\nStarting Evaluation Phase...")
        eval_rewards = []
        episodes_completed = 0
        
        while episodes_completed < self.MAX_EVAL_EPISODES:
            ep = self.MAX_TRAIN_EPISODES + 1000 + episodes_completed
            env = self.obelix_class(**self.env_kwargs, seed=self.args.seed + ep)
            s = env.reset(seed=self.args.seed + ep)
            s = np.asarray(s, dtype=np.float32)
            
            stack = deque(maxlen=self.args.frame_stack)
            for _ in range(self.args.frame_stack): stack.append(s.copy())
            
            ep_ret = 0.0
            for _ in range(self.args.max_steps):
                s_stacked = np.concatenate(stack).astype(np.float32)
                s_t = torch.from_numpy(s_stacked).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits = self.policyNetwork(s_t)
                    action = torch.argmax(logits, dim=-1).item()
                    
                s2, r, done = env.step(ACTIONS[action], render=False)
                s2 = np.asarray(s2, dtype=np.float32)
                ep_ret += float(r)
                
                s = s2
                stack.append(s2)
                if done: break
            
            eval_rewards.append(ep_ret)
            episodes_completed += 1
            
        mean_eval_score = np.mean(eval_rewards)
        std_eval_score = np.std(eval_rewards)
        
        self.performBookKeeping(train=False, ep_ret=mean_eval_score)
        print(f"Evaluation Complete! Mean Reward: {mean_eval_score:.1f} ± {std_eval_score:.1f}")
        return mean_eval_score, std_eval_score

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

# =====================================================================
# MAIN
# =====================================================================
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    OBELIX = import_obelix(args.obelix_py)
    env_kwargs = {
        "scaling_factor": args.scaling_factor,
        "arena_size": args.arena_size,
        "max_steps": args.max_steps,
        "wall_obstacles": args.wall_obstacles,
        "difficulty": args.difficulty,
        "box_speed": args.box_speed,
    }

    agent = VPG(obelix_class=OBELIX, env_kwargs=env_kwargs, args=args, device=device)
    
    plot_history, all_ep_rets, mean_eval, std_eval = agent.runVPG()
    
    print(f"\nTraining Complete! Best return: {agent.best_return:.1f}")
    checkpoint = {
        "state_dict": agent.policyNetwork.state_dict(),
        "value_state_dict": agent.valueNetwork.state_dict()
    }
    torch.save(checkpoint, args.out)
    print(f"Saved full VPG checkpoint → {args.out}")

    plotResults(plot_history, all_ep_rets, args)

if __name__ == "__main__":
    main()
