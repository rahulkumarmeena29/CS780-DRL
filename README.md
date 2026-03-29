# 🤖 CS780-DRL: Simulating OBELIX (Behaviour-based Robot)

![Teaser Image](./OBELIX.png)

## 📌 Overview

This repository implements a simulation of the **OBELIX robot**, inspired by the paper:

📄 *"Automatic Programming of Behaviour-based Robots using Reinforcement Learning"*
by **Sridhar Mahadevan** and **Jonathan Connell**

The project is part of reinforcement learning coursework and focuses on:

* Behaviour-based robotics
* Reinforcement Learning control
* Autonomous navigation

---

## 🎯 Features

* Manual robot control using keyboard
* Automatic control using RL algorithms
* Configurable environment difficulty
* Evaluation and leaderboard system on Codabench

---

## 🎮 Manual Gameplay

Run:

```bash
python manual_play.py
```

### Controls:

| Key | Action             |
| --- | ------------------ |
| W   | Move forward       |
| A   | Turn left (45°)    |
| Q   | Turn left (22.5°)  |
| E   | Turn right (22.5°) |
| D   | Turn right (45°)   |

---

## 🤖 Automatic Gameplay

Run:

```bash
python robot.py
```

### Example Code:

```python
import numpy as np
from obelix import OBELIX

bot = OBELIX(scaling_factor=5)

move_choice = ['L45', 'L22', 'FW', 'R22', 'R45']
user_input_choice = [ord("q"), ord("a"), ord("w"), ord("d"), ord("e")]

bot.render_frame()
episode_reward = 0

for step in range(1, 2000):
    random_step = np.random.choice(
        user_input_choice,
        1,
        p=[0.05, 0.1, 0.7, 0.1, 0.05]
    )[0]

    if random_step in user_input_choice:
        action = move_choice[user_input_choice.index(random_step)]

    sensor_feedback, reward, done = bot.step(action)
    episode_reward += reward

    print(step, sensor_feedback, episode_reward)
```

---

## 🧠 RL Evaluation Setup

### ✅ Success Condition

* Robot attaches to the box
* Episode ends when the box touches boundary

---

### ▶️ Run Evaluation

```bash
python evaluate.py \
  --agent_file agent_template.py \
  --runs 10 \
  --seed 0 \
  --max_steps 1000 \
  --wall_obstacles
```

---

## ⚙️ Difficulty Levels

| Level | Description           |
| ----- | --------------------- |
| 0     | Static box            |
| 2     | Blinking box          |
| 3     | Moving + blinking box |

Additional parameter:

```bash
--box_speed N
```

---

## 🧾 Submission

Edit:

```bash
agent_template.py
```

Implement:

```python
def policy(obs, rng) -> str:
    ...
```

### Valid Actions:

```
L45, L22, FW, R22, R45
```

---

## ⚠️ Known Issue

* Push feature is not fully implemented
* Current behavior = **attach instead of push**

---

## 📊 Leaderboard

* Results stored in `leaderboard.csv`
* Based on mean and standard deviation over multiple runs

---

## 🔗 References

* 📄 Mahadevan & Connell (1991)
* 🔗 https://cdn.aaai.org/AAAI/1991/AAAI91-120.pdf
* 🔗 https://github.com/iabhinavjoshi/OBELIX
* 🎓 NPTEL Reinforcement Learning Course (2023)

---

## 👨‍💻 Author

**Rahul Kumar Meena**

---

## ⭐ Repository

If you find this useful, consider giving it a ⭐ on GitHub!
