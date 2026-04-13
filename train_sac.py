from __future__ import annotations
import argparse, random, os, collections
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib.util

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)
OBS_DIM   = 18

# =====================================================================
# SUBPROCESS ENVIRONMENT WORKER & RND 
# =====================================================================
def _worker_fn(conn, obelix_py, env_kwargs, stuck_limit, seed_offset):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    stuck_counter = 0
    env = None

    while True:
        cmd, data = conn.recv()

        if cmd == "reset":
            ep_seed = data
            stuck_counter = 0
            env = OBELIX(**env_kwargs, seed=ep_seed)
            obs = env.reset(seed=ep_seed)
            conn.send(np.array(obs, dtype=np.float32))

        elif cmd == "step":
            action_str = data
            obs2, reward, done = env.step(action_str, render=False)
            obs2 = np.array(obs2, dtype=np.float32)

            if obs2[17] > 0.5:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter >= stuck_limit:
                done = True
                reward -= 100.0
                stuck_counter = 0

            conn.send((obs2, float(reward), bool(done)))

        elif cmd == "close":
            conn.close()
            return

class SubprocVecEnv:
    def __init__(self, obelix_py, num_envs, env_kwargs, stuck_limit=20, seed=0):
        self.num_envs    = num_envs
        self.seed        = seed
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        for i in range(num_envs):
            p = mp.Process(
                target=_worker_fn,
                args=(self.child_conns[i], obelix_py, env_kwargs, stuck_limit, seed + i),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

    def reset(self, episode: int = 0):
        for i, conn in enumerate(self.parent_conns):
            conn.send(("reset", self.seed + episode * self.num_envs + i))
        return np.stack([conn.recv() for conn in self.parent_conns])  

    def step(self, actions):
        for conn, a in zip(self.parent_conns, actions):
            conn.send(("step", ACTIONS[a]))
        results = [conn.recv() for conn in self.parent_conns]
        obs2, rewards, dones = zip(*results)
        return (np.stack(obs2), np.array(rewards), np.array(dones, dtype=bool))

    def close(self):
        for conn in self.parent_conns:
            conn.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)

class RNDTarget(nn.Module):
    def __init__(self, in_dim=OBS_DIM, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), 
            nn.LeakyReLU(0.1), 
            nn.Linear(128,128), 
            nn.LeakyReLU(0.1), 
            nn.Linear(128,out_dim)
        )
        for p in self.parameters(): p.requires_grad = False
    def forward(self, x): 
        return self.net(x)

class RNDPredictor(nn.Module):
    def __init__(self, in_dim=OBS_DIM, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), 
            nn.LeakyReLU(0.1), 
            nn.Linear(128,128), 
            nn.LeakyReLU(0.1), 
            nn.Linear(128,out_dim)
        )
    def forward(self, x): 
        return self.net(x)

class RND:
    def __init__(self, in_dim=OBS_DIM, out_dim=64, lr=1e-4, device="cpu", beta=0.1):
        self.target    = RNDTarget(in_dim, out_dim).to(device)
        self.predictor = RNDPredictor(in_dim, out_dim).to(device)
        self.opt       = optim.Adam(self.predictor.parameters(), lr=lr)
        self.device    = device
        self.beta      = beta
        self._mean, self._var, self._cnt = 0.0, 1.0, 1

    def intrinsic_reward(self, obs_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): t = self.target(obs_t)
        p = self.predictor(obs_t)
        return F.mse_loss(p, t.detach(), reduction="none").mean(dim=1)

    def update(self, obs_t: torch.Tensor):
        t = self.target(obs_t).detach()
        p = self.predictor(obs_t)
        loss = F.mse_loss(p, t)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def normalized(self, obs_t: torch.Tensor) -> torch.Tensor:
        raw = self.intrinsic_reward(obs_t)
        self._cnt += raw.numel()
        self._mean += (raw.mean().item() - self._mean) / self._cnt
        self._var   = max(self._var + ((raw - self._mean)**2).mean().item() / self._cnt, 1e-6)
        return self.beta * (raw - self._mean) / np.sqrt(self._var)

class ReplayBuffer:
    def __init__(self, bufferSize=300_000):
        self.buf = collections.deque(maxlen=bufferSize)
        
    def store(self, experience):
        s, a, r, s2, d = experience
        self.buf.append((s, int(a), float(r), s2, float(d)))
        
    def length(self): 
        return len(self.buf)
        
    def splitExperiences(self, experiences):
        s,a,r,s2,d = zip(*experiences)
        return (torch.tensor(np.array(s),dtype=torch.float32),
                torch.tensor(a,dtype=torch.long),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(np.array(s2),dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
                
    def sample(self, batchSize):
        return random.sample(self.buf, batchSize)

# =====================================================================
# SAC NETWORKS
# =====================================================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM,256),
            nn.ReLU(), 
            nn.Linear(256,256), 
            nn.ReLU(), 
            nn.Linear(256,N_ACTIONS)
        )
    def forward(self, x):
        probs = F.softmax(self.net(x), dim=-1)
        return probs, F.log_softmax(self.net(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM,256), 
            nn.ReLU(), 
            nn.Linear(256,256), 
            nn.ReLU(), 
            nn.Linear(256,N_ACTIONS)
        )
    def forward(self, x): 
        return self.net(x)

# =====================================================================
# SAC AGENT
# =====================================================================
class SAC:
    def __init__(self, env, gamma, tau, bufferSize, entropyLR, updateFrequency, 
                 policyOptimizerLR, valueOptimizerLR, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES, 
                 args, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.updateFrequency = updateFrequency
        self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
        self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
        self.args = args
        self.device = device
        
        # Initializing Networks
        self.policyNetwork = PolicyNetwork().to(device)
        self.onlineValueNetwork_1 = ValueNetwork().to(device)
        self.onlineValueNetwork_2 = ValueNetwork().to(device)
        self.targetValueNetwork_1 = ValueNetwork().to(device)
        self.targetValueNetwork_2 = ValueNetwork().to(device)
        
        self.targetValueNetwork_1.load_state_dict(self.onlineValueNetwork_1.state_dict())
        self.targetValueNetwork_2.load_state_dict(self.onlineValueNetwork_2.state_dict())

        # Entropy Tuning
        self.target_entropy = -np.log(1.0 / N_ACTIONS) * 0.5
        self.logAlpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.logAlpha.exp().item()

        # Load full agent state if curriculum learning is active
        if args.load_weights:
            checkpoint = torch.load(args.load_weights, map_location=device)
            
            # Check if it's our full checkpoint format
            if isinstance(checkpoint, dict) and "q1_state_dict" in checkpoint:
                self.policyNetwork.load_state_dict(checkpoint["state_dict"])
                self.onlineValueNetwork_1.load_state_dict(checkpoint["q1_state_dict"])
                self.onlineValueNetwork_2.load_state_dict(checkpoint["q2_state_dict"])
                self.targetValueNetwork_1.load_state_dict(checkpoint["target_q1_state_dict"])
                self.targetValueNetwork_2.load_state_dict(checkpoint["target_q2_state_dict"])
                self.logAlpha.data = checkpoint["log_alpha"]
                self.alpha = self.logAlpha.exp().item()
                print(f"Loaded FULL agent checkpoint from {args.load_weights} (Curriculum mode active)")
            else:
                # Fallback just in case you load an old policy-only weights file
                sd = checkpoint["state_dict"] if (isinstance(checkpoint, dict) and "state_dict" in checkpoint) else checkpoint
                self.policyNetwork.load_state_dict(sd, strict=True)
                print(f"Warning: Loaded ONLY actor weights from {args.load_weights}. Critics are randomly initialized.")

        # Optimizers 
        self.policyOptimizerFn = optim.Adam(self.policyNetwork.parameters(), lr=args.lr)
        self.valueOptimizerFn_1 = optim.Adam(self.onlineValueNetwork_1.parameters(), lr=args.lr)
        self.valueOptimizerFn_2 = optim.Adam(self.onlineValueNetwork_2.parameters(), lr=args.lr)
        self.alphaOptimizerFn = optim.Adam([self.logAlpha], lr=args.lr)

        # Buffer & RND
        self.rBuffer = ReplayBuffer(bufferSize)
        self.rnd = RND(beta=args.rnd_beta, device=device)

        self.initBookKeeping()

    def save_checkpoint(self, path):
        # Saves the ENTIRE agent state so training can resume flawlessly
        checkpoint = {
            "state_dict": self.policyNetwork.state_dict(),
            "q1_state_dict": self.onlineValueNetwork_1.state_dict(),
            "q2_state_dict": self.onlineValueNetwork_2.state_dict(),
            "target_q1_state_dict": self.targetValueNetwork_1.state_dict(),
            "target_q2_state_dict": self.targetValueNetwork_2.state_dict(),
            "log_alpha": self.logAlpha.data
        }
        torch.save(checkpoint, path)

    def initBookKeeping(self):
        self.n_done = 0
        self.total_success = 0
        self.best_return = -float("inf")
        self.plot_history = []
        self.all_ep_rets = []
        self.last50_ret = []
        self.last50_suc = []
        self.ep_rets = np.zeros(self.env.num_envs)

    def selectRandomAction(self):
        return np.array([random.randint(0, N_ACTIONS-1) for _ in range(self.env.num_envs)])
        
    def selectAction(self, obs_batch):
        with torch.no_grad():
            x = torch.tensor(obs_batch, dtype=torch.float32).to(self.device)
            probs, _ = self.policyNetwork(x)
            return torch.distributions.Categorical(probs).sample().cpu().numpy()

    def runSAC(self):
        plot_history, all_ep_rets = self.trainAgent()        
        mean_eval, std_eval = self.evaluateAgent()     
        return plot_history, all_ep_rets, mean_eval, std_eval

    def trainAgent(self):
        obs = self.env.reset(episode=0)
        last_logged = -1

        while self.n_done < self.MAX_TRAIN_EPISODES:
            # Action Selection
            if self.rBuffer.length() < self.args.replay_start:
                actions = self.selectRandomAction()
            else:
                actions = self.selectAction(obs)

            obs2, rewards, dones = self.env.step(actions)

            # Step Processing 
            for i in range(self.env.num_envs):
                experience = (obs[i], actions[i], rewards[i], obs2[i], dones[i])
                self.rBuffer.store(experience)
                self.ep_rets[i] += rewards[i]

                if dones[i]:
                    success = int(rewards[i] > 1000.0)
                    self.performBookKeeping(train=True, idx=i, ep_ret=self.ep_rets[i], success=success)
                    self.ep_rets[i] = 0

                    obs2[i] = self.env.parent_conns[i].recv() if False else obs2[i] 
                    self.env.parent_conns[i].send(("reset", self.args.seed + self.n_done + i))
                    obs2[i] = self.env.parent_conns[i].recv()

            obs = obs2

            # Network Training
            if self.rBuffer.length() >= self.args.replay_start:
                for _ in range(self.args.updates_per_step):
                    experiences = self.rBuffer.sample(self.args.batch)
                    self.trainNetworks(experiences)
                # Removed double updateValueNetworks call here!

            # Logging
            if self.n_done > 0 and self.n_done % 50 == 0 and self.n_done != last_logged and len(self.last50_ret) > 0:
                last_logged = self.n_done
                mean_50 = np.mean(self.last50_ret)
                self.plot_history.append((self.n_done, mean_50))
                print(
                    f"Ep {self.n_done}/{self.MAX_TRAIN_EPISODES} "
                    f"mean={mean_50:.1f} "
                    f"max={np.max(self.last50_ret):.1f} "
                    f"succ_50={int(np.sum(self.last50_suc))} "
                    f"alpha={self.alpha:.3f} "
                    f"buf={self.rBuffer.length()} "
                    f"best={self.best_return:.1f}"
                )

        return self.plot_history, self.all_ep_rets

    def trainNetworks(self, experiences):
        s, a, r, s2, d = self.rBuffer.splitExperiences(experiences)

        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)
        
        # RND Integration
        r_int = self.rnd.normalized(s2)
        self.rnd.update(s2)
        r_total = r + r_int.detach()

        # Target Q Values
        with torch.no_grad():
            p2, lp2 = self.policyNetwork(s2)
            minq = torch.min(self.targetValueNetwork_1(s2), self.targetValueNetwork_2(s2))
            v_next = (p2 * (minq - self.alpha * lp2)).sum(1)
            tq = r_total + self.gamma * (1-d) * v_next

        # Value Network Updates
        q1 = self.onlineValueNetwork_1(s).gather(1,a.unsqueeze(1)).squeeze()
        q2 = self.onlineValueNetwork_2(s).gather(1,a.unsqueeze(1)).squeeze()
        
        q1_loss = F.mse_loss(q1, tq)
        self.valueOptimizerFn_1.zero_grad()
        q1_loss.backward()
        self.valueOptimizerFn_1.step()

        q2_loss = F.mse_loss(q2, tq)
        self.valueOptimizerFn_2.zero_grad()
        q2_loss.backward()
        self.valueOptimizerFn_2.step()

        # Policy Network Update
        p, lp = self.policyNetwork(s)
        minq_s = torch.min(self.onlineValueNetwork_1(s), self.onlineValueNetwork_2(s))
        policyLoss = (p * (self.alpha * lp - minq_s)).sum(1).mean()
        
        self.policyOptimizerFn.zero_grad()
        policyLoss.backward()
        self.policyOptimizerFn.step()

        # Entropy Weight Update
        entropy = -(p * lp).sum(1).detach()
        alphaLoss = (self.logAlpha * (entropy - self.target_entropy)).mean()
        
        self.alphaOptimizerFn.zero_grad()
        alphaLoss.backward()
        self.alphaOptimizerFn.step()
        
        self.logAlpha.data.clamp_(-3.0, 0.5)
        self.alpha = self.logAlpha.exp().item()

        # Soft Update Targets
        self.updateValueNetworks(self.tau)

    def updateValueNetworks(self, tau):
        for tParam, oParam in zip(self.targetValueNetwork_1.parameters(), self.onlineValueNetwork_1.parameters()):
            mixedWeights = (1 - tau) * tParam.data + tau * oParam.data
            tParam.data.copy_(mixedWeights)
            
        for tParam, oParam in zip(self.targetValueNetwork_2.parameters(), self.onlineValueNetwork_2.parameters()):
            mixedWeights = (1 - tau) * tParam.data + tau * oParam.data
            tParam.data.copy_(mixedWeights)

    def performBookKeeping(self, train=True, idx=0, ep_ret=0, success=0):
        if train:
            self.last50_ret.append(ep_ret)
            self.last50_suc.append(success)
            if len(self.last50_ret) > 50: 
                self.last50_ret.pop(0)
                self.last50_suc.pop(0)
                
            if ep_ret > self.best_return:
                self.best_return = ep_ret
                self.save_checkpoint(self.args.out) # Full checkpoint save
                
            self.total_success += success
            self.all_ep_rets.append((self.n_done, ep_ret))
            self.n_done += 1
        else:
            self.final_eval_score = ep_ret

    def evaluateAgent(self):
        print("\nStarting Evaluation Phase...")
        eval_rewards = []
        obs = self.env.reset(episode=self.MAX_TRAIN_EPISODES + 1)
        current_ep_rewards = np.zeros(self.env.num_envs)
        episodes_completed = 0

        while episodes_completed < self.MAX_EVAL_EPISODES:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).to(self.device)
                probs, _ = self.policyNetwork(x)
                actions = torch.argmax(probs, dim=-1).cpu().numpy()

            obs2, rewards, dones = self.env.step(actions)

            for i in range(self.env.num_envs):
                current_ep_rewards[i] += rewards[i]
                
                if dones[i]:
                    eval_rewards.append(current_ep_rewards[i])
                    current_ep_rewards[i] = 0
                    episodes_completed += 1
                    
                    obs2[i] = self.env.parent_conns[i].recv() if False else obs2[i] 
                    self.env.parent_conns[i].send(("reset", self.args.seed + 10000 + episodes_completed))
                    obs2[i] = self.env.parent_conns[i].recv()
                    
                    if episodes_completed >= self.MAX_EVAL_EPISODES:
                        break

            obs = obs2
            
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
        ax.set_title(f"SAC Training — D{args.difficulty} {'+ walls' if args.wall_obstacles else ''}")
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
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser(description="Vectorized SAC+RND for OBELIX")
    ap.add_argument("--obelix_py",        type=str,   required=True)
    ap.add_argument("--out",              type=str,   default="sac_weights.pth")
    ap.add_argument("--episodes",         type=int,   default=2000)
    ap.add_argument("--max_steps",        type=int,   default=1000)
    ap.add_argument("--num_envs",         type=int,   default=8)
    ap.add_argument("--difficulty",       type=int,   default=0)
    ap.add_argument("--wall_obstacles",   action="store_true")
    ap.add_argument("--box_speed",        type=int,   default=2)
    ap.add_argument("--scaling_factor",   type=int,   default=5)
    ap.add_argument("--arena_size",       type=int,   default=500)
    ap.add_argument("--gamma",            type=float, default=0.99)
    ap.add_argument("--lr",               type=float, default=3e-4)
    ap.add_argument("--batch",            type=int,   default=512)
    ap.add_argument("--replay_start",     type=int,   default=5000)
    ap.add_argument("--updates_per_step", type=int,   default=2)
    ap.add_argument("--rnd_beta",         type=float, default=0.1)
    ap.add_argument("--rnd_beta_end",     type=float, default=0.02)
    ap.add_argument("--load_weights",     type=str,   default="")
    ap.add_argument("--seed",             type=int,   default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Parallel envs: {args.num_envs}")

    env_kwargs = dict(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
    )
    sac_env = SubprocVecEnv(args.obelix_py, args.num_envs, env_kwargs, seed=args.seed)

    agent = SAC(
        env=sac_env,
        gamma=args.gamma,
        tau=0.005,
        bufferSize=300_000,
        entropyLR=args.lr,
        updateFrequency=1,
        policyOptimizerLR=args.lr,
        valueOptimizerLR=args.lr,
        MAX_TRAIN_EPISODES=args.episodes,
        MAX_EVAL_EPISODES=10,
        args=args,
        device=device
    )    
    plot_history, all_ep_rets, mean_eval, std_eval = agent.runSAC()

    sac_env.close()
    print(f"\nDone! Successes: {agent.total_success}/{args.episodes}")
    agent.save_checkpoint(args.out)
    print(f"Saved full agent checkpoint → {args.out}")

    plotResults(plot_history, all_ep_rets, args)

if __name__ == "__main__":
    main()
