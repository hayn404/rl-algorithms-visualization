import modal

app = modal.App("dqn-train")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gymnasium[atari,accept-rom-license]", "ale-py", "shimmy[atari]", "torch", "numpy", "pillow"
)


@app.function(image=image, gpu="T4", timeout=3600)
def train(env_name, num_episodes, epsilon, gamma, batch_size):
    import gymnasium as gym
    import ale_py
    import shimmy
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import random
    from collections import deque
    from PIL import Image
    import io

    device = torch.device("cuda")

    class DQN(nn.Module):
        def __init__(self, n_actions):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 32, 8, 4)
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            self.fc1 = nn.Linear(3136, 512)
            self.fc2 = nn.Linear(512, n_actions)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    def preprocess(frame):
        img = Image.fromarray(frame).convert('L').resize((84, 84))
        return np.array(img, dtype=np.float32) / 255.0

    env = gym.make(f"ALE/{env_name}-v5", render_mode="rgb_array")
    n_actions = env.action_space.n
    model = DQN(n_actions).to(device)
    target = DQN(n_actions).to(device)
    target.load_state_dict(model.state_dict())
    opt = optim.Adam(model.parameters(), lr=0.0001)
    buffer = deque(maxlen=50000)

    rewards = []
    lengths = []

    for ep in range(num_episodes):
        frame, _ = env.reset()
        frames = deque([preprocess(frame)] * 4, maxlen=4)
        state = np.stack(frames)
        total_reward = 0
        steps = 0
        eps = max(0.01, epsilon * (1 - ep / num_episodes))

        for _ in range(10000):
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = model(torch.FloatTensor(state).unsqueeze(0).to(device))
                action = q.argmax().item()

            next_frame, reward, done, trunc, _ = env.step(action)
            frames.append(preprocess(next_frame))
            next_state = np.stack(frames)
            buffer.append((state, action, reward, next_state, done or trunc))
            total_reward += reward
            steps += 1
            state = next_state

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                s, a, r, ns, d = zip(*batch)
                s = torch.FloatTensor(np.array(s)).to(device)
                a = torch.LongTensor(a).to(device)
                r = torch.FloatTensor(r).to(device)
                ns = torch.FloatTensor(np.array(ns)).to(device)
                d = torch.FloatTensor(d).to(device)

                opt.zero_grad()
                q_val = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target(ns).max(1)[0]
                    target_q = r + gamma * next_q * (1 - d)
                loss = nn.MSELoss()(q_val, target_q)
                loss.backward()
                opt.step()

            if done or trunc:
                break

        if (ep + 1) % 10 == 0:
            target.load_state_dict(model.state_dict())

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {ep+1}/{num_episodes}: Reward={total_reward}")

    env.close()

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return {"model": buf.getvalue(), "rewards": rewards, "lengths": lengths}
