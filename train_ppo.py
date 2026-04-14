from __future__ import annotations
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int = 5):
        super().__init__()
        self.net = nn.ModuleList([
            layer_init(nn.Linear(in_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, n_actions), std=0.01),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.net[1](self.net[0](x))
        x2 = self.net[3](self.net[2](x1)) + x1
        return self.net[4](x2)

class ValueNetwork(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.ModuleList([
            layer_init(nn.Linear(in_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.net[1](self.net[0](x))
        x2 = self.net[3](self.net[2](x1)) + x1
        return self.net[4](x2).squeeze(-1)


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


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

class EpisodeBuffer:
    def __init__(self):
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_old_log_probs = []
        self.buffer_returns = []
        self.buffer_advantages = []
        self.buffer_values = []
        self.batch_episode_returns = []
        self.batch_successes = 0
        
    def fill(self, states, actions, log_probs_old, returns, advantages, values, ep_ret_raw, is_success):
        self.buffer_states.extend(states)
        self.buffer_actions.extend(actions)
        self.buffer_old_log_probs.extend(log_probs_old)
        self.buffer_returns.extend(returns.tolist())
        self.buffer_advantages.extend(advantages.tolist())
        self.buffer_values.extend(values[:-1])
        self.batch_episode_returns.append(ep_ret_raw)
        if is_success:
            self.batch_successes += 1
    
    def returnElements(self):
        return (self.buffer_states, self.buffer_actions, self.buffer_old_log_probs, 
                self.buffer_returns, self.buffer_advantages, self.buffer_values)
                
    def reset(self):
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_old_log_probs = []
        self.buffer_returns = []
        self.buffer_advantages = []
        self.buffer_values = []
        self.batch_episode_returns = []
        self.batch_successes = 0

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

class PPO:
    def __init__(self, args, device, OBELIX):
        self.args = args
        self.device = device
        self.OBELIX = OBELIX
        self.policy_net = PolicyNetwork(in_dim=18).to(device)
        
        if args.load_weights != "":
            print(f"Loading pre-trained policy weights from {args.load_weights}...")
            sd = torch.load(args.load_weights, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            self.policy_net.load_state_dict(sd, strict=False)
        else:
            self.policy_net.net[-1].bias.data[2] += 1.0

        self.teacher_net = None
        if args.teacher_weights != "":
            print(f"Loading teacher policy for RLHF from {args.teacher_weights}...")
            self.teacher_net = PolicyNetwork(in_dim=18).to(device)
            sd_teacher = torch.load(args.teacher_weights, map_location=device)
            if isinstance(sd_teacher, dict) and "state_dict" in sd_teacher:
                sd_teacher = sd_teacher["state_dict"]
            self.teacher_net.load_state_dict(sd_teacher, strict=False)
            self.teacher_net.eval()

        self.value_net = ValueNetwork(in_dim=18).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.value_lr)
        
        self.rBuffer = EpisodeBuffer()
        
        self.best_raw_return = -float("inf")
        self.best_state_dict = None
        self.best_episode_num = 0
        self.total_episodes_done = 0
        self.cumulative_return = 0.0
        self.total_successes = 0

    def runPPO(self):
        self.trainAgent()
        print(f"\nTraining Complete!")
        print(f"Total Successes: {self.total_successes}/{self.total_episodes_done}")
        
        torch.save(self.policy_net.state_dict(), self.args.out)
        print("Saved final trained policy to:", self.args.out)
        
        if self.best_state_dict is not None:
            torch.save({
                "episode": self.best_episode_num,
                "state_dict": self.best_state_dict,
                "best_raw_return": self.best_raw_return,
            }, self.args.best_out)
            print("Saved anomalous 'best single episode' checkpoint to:", self.args.best_out)

    def trainAgent(self):
        while self.total_episodes_done < self.args.episodes:
            self.rBuffer.reset()
            
            frac = max(0.0, 1.0 - (self.total_episodes_done / self.args.episodes))
            cur_policy_lr = self.args.policy_lr * frac
            cur_value_lr = self.args.value_lr * frac
            for param_group in self.policy_optimizer.param_groups:
                param_group["lr"] = cur_policy_lr
            for param_group in self.value_optimizer.param_groups:
                param_group["lr"] = cur_value_lr
            
            self.cur_entropy_coef = max(0.01, self.args.entropy_coef * frac)
            self.cur_clip_eps = max(0.05, self.args.clip_eps * frac)
            
            episodes_this_round = min(self.args.episodes_per_update, self.args.episodes - self.total_episodes_done)
            
            for ep_local in range(episodes_this_round):
                ep = self.total_episodes_done + ep_local
                env = self.OBELIX(
                    scaling_factor=self.args.scaling_factor,
                    arena_size=self.args.arena_size,
                    max_steps=self.args.max_steps,
                    wall_obstacles=self.args.wall_obstacles,
                    difficulty=self.args.difficulty,
                    box_speed=self.args.box_speed,
                    seed=self.args.seed + ep,
                )
                s = env.reset(seed=self.args.seed + ep)
                s = np.asarray(s, dtype=np.float32)
                
                states = []
                actions = []
                log_probs_old = []
                rewards = []
                dones = []
                values = []
                ep_ret_raw = 0.0
                is_success = False
                stuck_counter = 0
                
                for _ in range(self.args.max_steps):
                    s_t = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = self.policy_net(s_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                        value = self.value_net(s_t)

                    act_idx = action.item()
                    s2, r, done = env.step(ACTIONS[act_idx], render=False)
                    s2 = np.asarray(s2, dtype=np.float32)
                    
                    if s2[17] > 0.5:
                        stuck_counter += 1
                    else:
                        stuck_counter = 0
                        
                    ep_ret_raw += float(r)  
                    
                    if float(r) > 1000.0:
                        is_success = True
                    
                    states.append(s.copy())
                    actions.append(act_idx)
                    log_probs_old.append(log_prob.item())
                    rewards.append(float(r))
                    dones.append(float(done))
                    values.append(value.item())
                    
                    s = s2
                    if done:
                        break

                s_t = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    last_value = self.value_net(s_t).item()
                values.append(last_value)
                
                if len(rewards) == 0:
                    continue

                advantages, returns = compute_gae(
                    rewards=np.array(rewards, dtype=np.float32),
                    values=np.array(values, dtype=np.float32),
                    dones=np.array(dones, dtype=np.float32),
                    gamma=self.args.gamma,
                    lam=self.args.gae_lambda,
                )
                
                self.rBuffer.fill(states, actions, log_probs_old, returns, advantages, values, ep_ret_raw, is_success)
                self.cumulative_return += ep_ret_raw
                if is_success:
                    self.total_successes += 1
                if ep_ret_raw > self.best_raw_return:
                    self.best_raw_return = ep_ret_raw
                    self.best_state_dict = copy.deepcopy(self.policy_net.state_dict())
                    self.best_episode_num = ep
                    
            prev_total = self.total_episodes_done
            self.total_episodes_done += episodes_this_round

            if len(self.rBuffer.buffer_states) == 0:
                continue
                
            self.trainNetworks()

            if prev_total // 50 < self.total_episodes_done // 50 or self.total_episodes_done == self.args.episodes:
                mean_ret = float(np.mean(self.rBuffer.batch_episode_returns)) if self.rBuffer.batch_episode_returns else float("nan")
                max_ret = float(np.max(self.rBuffer.batch_episode_returns)) if self.rBuffer.batch_episode_returns else float("nan")
                min_ret = float(np.min(self.rBuffer.batch_episode_returns)) if self.rBuffer.batch_episode_returns else float("nan")

                print(
                    f"Episodes {self.total_episodes_done - episodes_this_round + 1}-{self.total_episodes_done}/{self.args.episodes} "
                    f"mean_return={mean_ret:.2f} "
                    f"max_return={max_ret:.2f} "
                    f"min_return={min_ret:.2f} "
                    f"policy_loss={self.last_policy_loss:.4f} "
                    f"value_loss={self.last_value_loss:.4f} "
                    f"best_return={self.best_raw_return:.2f}"
                )

    def trainNetworks(self):
        buffer_states, buffer_actions, buffer_old_log_probs, buffer_returns, buffer_advantages, buffer_values = self.rBuffer.returnElements()
        
        states_t = torch.tensor(np.array(buffer_states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(buffer_actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(buffer_old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(buffer_returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(buffer_advantages, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(buffer_values, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        n_samples = len(buffer_states)

        for epoch in range(self.args.ppo_epochs):
            epoch_kls = []
            for idx in minibatch_indices(n_samples, self.args.batch_size):
                batch_states = states_t[idx]
                batch_actions = actions_t[idx]
                batch_old_log_probs = old_log_probs_t[idx]
                batch_returns = returns_t[idx]
                batch_advantages = advantages_t[idx]
                batch_values = values_t[idx]
                logits = self.policy_net(batch_states)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logits = torch.zeros_like(logits)
                logits = torch.clamp(logits, -20.0, 20.0)
                
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                epoch_kls.append(approx_kl.item())
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cur_clip_eps, 1.0 + self.cur_clip_eps) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean() - self.cur_entropy_coef * entropy
                
                values_pred = self.value_net(batch_states)
                value_loss = 0.5 * ((values_pred - batch_returns) ** 2).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.args.max_grad_norm)
                self.policy_optimizer.step()
                
                self.value_optimizer.zero_grad()
                (self.args.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.args.max_grad_norm)
                self.value_optimizer.step()

                self.last_policy_loss = policy_loss.item()
                self.last_value_loss = value_loss.item()

            mean_kl = float(np.mean(epoch_kls))
            if mean_kl > 1.5 * self.args.target_kl:
                break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--load_weights", type=str, default="", help="Path to pre-trained policy weights")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--episodes_per_update", type=int, default=64) 
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.90)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--target_kl", type=float, default=0.015)
    ap.add_argument("--policy_lr", type=float, default=1e-5)
    ap.add_argument("--value_lr", type=float, default=5e-4)
    ap.add_argument("--entropy_coef", type=float, default=0.01) 
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--ppo_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--reward_scale", type=float, default=0.005)
    ap.add_argument("--teacher_weights", type=str, default="", help="Path to teacher policy for RLHF KL penalty")
    ap.add_argument("--kl_beta", type=float, default=0.01, help="KL penalty coefficient for RLHF")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best_out", type=str, default="best_policy.pth",
                    help="Path to save the richest best-policy checkpoint (default: best_policy.pth)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    OBELIX = import_obelix(args.obelix_py)
    
    ppo_agent = PPO(args, device, OBELIX)
    ppo_agent.runPPO()

if __name__ == "__main__":
    main()
