from __future__ import annotations
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class PolicyNetwork(nn.Module):
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

class ValueNetwork(nn.Module):
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


class VPG:
    def __init__(self, args, device, OBELIX):
        self.args = args
        self.device = device
        self.OBELIX = OBELIX
        
        self.policy_net = PolicyNetwork().to(device)
        self.value_net = ValueNetwork().to(device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.value_lr)
        
        self.initBookKeeping()

    def initBookKeeping(self):
        self.total_episodes_done = 0
        self.best_raw_return = -float('inf')
        self.best_state_dict = None
        
        # Trackers for the current episode's data
        self.ep_states = []
        self.ep_log_probs = []
        self.ep_rewards = []
        self.ep_entropies = []
        
        # Trackers for logging
        self.current_ep_return = 0.0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0

    def runVPG(self):
        self.trainAgent()
        
        print("\nTraining Complete! Running final evaluation...")
        eval_mean, eval_std = self.evaluateAgent()
        print(f"Final Eval Score: {eval_mean:.2f} ± {eval_std:.2f}")
        
        torch.save(self.policy_net.state_dict(), self.args.out)
        print("Saved final policy to:", self.args.out)
        
        if self.best_state_dict is not None:
            best_out = self.args.out.replace(".pth", "_best.pth")
            torch.save(self.best_state_dict, best_out)
            print("Saved best policy to:", best_out)

    def trainAgent(self):
        while self.total_episodes_done < self.args.episodes:
            env = self.OBELIX(
                scaling_factor=self.args.scaling_factor,
                arena_size=self.args.arena_size,
                max_steps=self.args.max_steps,
                wall_obstacles=self.args.wall_obstacles,
                difficulty=self.args.difficulty,
                box_speed=self.args.box_speed,
                seed=self.args.seed + self.total_episodes_done,
            )
            s = env.reset(seed=self.args.seed + self.total_episodes_done)
            s = np.asarray(s, dtype=np.float32)
            
            # Reset episode trackers
            self.ep_states = []
            self.ep_log_probs = []
            self.ep_rewards = []
            self.ep_entropies = []
            self.current_ep_return = 0.0

            for _ in range(self.args.max_steps):
                s_t = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                logits = self.policy_net(s_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                act_idx = action.item()

                s2, r, done = env.step(ACTIONS[act_idx], render=False)
                s2 = np.asarray(s2, dtype=np.float32)
                
                self.ep_log_probs.append(dist.log_prob(action))
                self.ep_entropies.append(dist.entropy())
                self.ep_states.append(s.copy())
                self.ep_rewards.append(float(r))
                
                self.current_ep_return += float(r)
                s = s2
                
                if done:
                    break

            if len(self.ep_rewards) > 0:
                self.trainNetworks()
            
            self.total_episodes_done += 1
            self.performBookKeeping(train=True)

    def trainNetworks(self):
        returns = compute_returns(self.ep_rewards, self.args.gamma)
        returns_t = torch.from_numpy(returns).float().to(self.device)
        
        if self.args.normalize_returns and len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        states_t = torch.from_numpy(np.stack(self.ep_states)).float().to(self.device)
        values_t = self.value_net(states_t)
        
        advantages_t = returns_t - values_t.detach()
        if self.args.normalize_advantages and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        log_probs_t = torch.cat(self.ep_log_probs)
        entropies_t = torch.cat(self.ep_entropies)
        
        policy_loss = -(log_probs_t * advantages_t).mean() - self.args.entropy_coef * entropies_t.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.policy_optimizer.step()
        
        value_loss = nn.functional.smooth_l1_loss(values_t, returns_t, beta=0.5)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 5.0)
        self.value_optimizer.step()

        # Update bookkeeping variables
        self.last_policy_loss = policy_loss.item()
        self.last_value_loss = value_loss.item()

    def evaluateAgent(self):
        rewards = []
        # Turn on eval mode for deterministic behavior if you add dropout/batchnorm later
        self.policy_net.eval() 
        
        for e in range(self.args.max_eval_episodes):
            env = self.OBELIX(
                scaling_factor=self.args.scaling_factor,
                arena_size=self.args.arena_size,
                max_steps=self.args.max_steps,
                wall_obstacles=self.args.wall_obstacles,
                difficulty=self.args.difficulty,
                box_speed=self.args.box_speed,
                seed=self.args.seed + 10000 + e, # Offset seed for evaluation
            )
            s = env.reset()
            s = np.asarray(s, dtype=np.float32)
            ep_rs = 0.0
            
            for _ in range(self.args.max_steps):
                s_t = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.policy_net(s_t)
                    # Greedy action selection for evaluation
                    act_idx = torch.argmax(logits).item() 

                s2, r, done = env.step(ACTIONS[act_idx], render=False)
                ep_rs += float(r)
                s = np.asarray(s2, dtype=np.float32)
                
                if done:
                    break
                    
            rewards.append(ep_rs)
            
        self.policy_net.train() # Turn training mode back on
        self.performBookKeeping(train=False)
        return float(np.mean(rewards)), float(np.std(rewards))

    def performBookKeeping(self, train=True):
        if train:
            # Track best models
            if self.current_ep_return > self.best_raw_return:
                self.best_raw_return = self.current_ep_return
                self.best_state_dict = copy.deepcopy(self.policy_net.state_dict())
                
            # Print periodic updates
            if self.total_episodes_done % 50 == 0:
                print(
                    f"Episode {self.total_episodes_done}/{self.args.episodes} "
                    f"return={self.current_ep_return:.2f} "
                    f"policy_loss={self.last_policy_loss:.4f} "
                    f"value_loss={self.last_value_loss:.4f} "
                    f"best_return={self.best_raw_return:.2f}"
                )
        else:
            # You can add evaluation-specific logging here if desired
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--max_eval_episodes", type=int, default=10) # Added for evaluateAgent()
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
    
    vpg_agent = VPG(args, device, OBELIX)
    vpg_agent.runVPG()

if __name__ == "__main__":
    main()
