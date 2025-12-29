# RL Algorithms Visualization
import streamlit as st
import gymnasium as gym
import ale_py
import shimmy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image
import time
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os

SAVE_DIR = "saved_weights"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.set_page_config(page_title="RL Algorithms Visualization", layout="wide")

# Q-table stored as dictionary
if 'Q' not in st.session_state:
    st.session_state.Q = {}
if 'V' not in st.session_state:
    st.session_state.V = {}
if 'rewards' not in st.session_state:
    st.session_state.rewards = []
if 'lengths' not in st.session_state:
    st.session_state.lengths = []
if 'episode' not in st.session_state:
    st.session_state.episode = 0

# DQN model storage
if 'dqn_model' not in st.session_state:
    st.session_state.dqn_model = None
if 'dqn_target' not in st.session_state:
    st.session_state.dqn_target = None
if 'dqn_optimizer' not in st.session_state:
    st.session_state.dqn_optimizer = None

# training control
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'training' not in st.session_state:
    st.session_state.training = False
if 'replay_buffer' not in st.session_state:
    st.session_state.replay_buffer = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# save weights to file
def save_weights():
    data = {
        'Q': st.session_state.Q,
        'V': st.session_state.V,
        'rewards': st.session_state.rewards,
        'lengths': st.session_state.lengths,
        'episode': st.session_state.episode,
    }
    if 'mc_counts' in st.session_state:
        data['mc_counts'] = st.session_state.mc_counts
    if 'mc_sums' in st.session_state:
        data['mc_sums'] = st.session_state.mc_sums

    with open(os.path.join(SAVE_DIR, "q_table.pkl"), "wb") as f:
        pickle.dump(data, f)

    # save DQN model if exists
    if st.session_state.dqn_model is not None:
        torch.save(st.session_state.dqn_model.state_dict(), os.path.join(SAVE_DIR, "dqn_model.pt"))
        torch.save(st.session_state.dqn_target.state_dict(), os.path.join(SAVE_DIR, "dqn_target.pt"))


# load weights from file
def load_weights():
    q_path = os.path.join(SAVE_DIR, "q_table.pkl")
    if os.path.exists(q_path):
        with open(q_path, "rb") as f:
            data = pickle.load(f)
        st.session_state.Q = data.get('Q', {})
        st.session_state.V = data.get('V', {})
        st.session_state.rewards = data.get('rewards', [])
        st.session_state.lengths = data.get('lengths', [])
        st.session_state.episode = data.get('episode', 0)
        if 'mc_counts' in data:
            st.session_state.mc_counts = data['mc_counts']
        if 'mc_sums' in data:
            st.session_state.mc_sums = data['mc_sums']


# delete saved weights
def delete_weights():
    q_path = os.path.join(SAVE_DIR, "q_table.pkl")
    dqn_path = os.path.join(SAVE_DIR, "dqn_model.pt")
    target_path = os.path.join(SAVE_DIR, "dqn_target.pt")
    if os.path.exists(q_path):
        os.remove(q_path)
    if os.path.exists(dqn_path):
        os.remove(dqn_path)
    if os.path.exists(target_path):
        os.remove(target_path)


# auto-load on startup
if 'loaded' not in st.session_state:
    load_weights()
    st.session_state.loaded = True


# helper to get Q values for a state
def get_q(state, n_actions):
    if state not in st.session_state.Q:
        st.session_state.Q[state] = np.zeros(n_actions)
    # if array is too small, expand it
    elif len(st.session_state.Q[state]) < n_actions:
        old_q = st.session_state.Q[state]
        st.session_state.Q[state] = np.zeros(n_actions)
        st.session_state.Q[state][:len(old_q)] = old_q
    return st.session_state.Q[state]

def get_v(state):
    if state not in st.session_state.V:
        st.session_state.V[state] = 0.0
    return st.session_state.V[state]


# epsilon greedy action selection
def choose_action(state, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        q_vals = get_q(state, n_actions)
        return np.argmax(q_vals)


# Q-Learning update
def qlearning_update(state, action, reward, next_state, done, n_actions, alpha, gamma):
    q_vals = get_q(state, n_actions)
    next_q = get_q(next_state, n_actions)

    if done:
        target = reward
    else:
        target = reward + gamma * np.max(next_q)

    td_error = target - q_vals[action]
    q_vals[action] += alpha * td_error
    st.session_state.Q[state] = q_vals


# SARSA update
def sarsa_update(state, action, reward, next_state, next_action, done, n_actions, alpha, gamma):
    q_vals = get_q(state, n_actions)
    next_q = get_q(next_state, n_actions)

    if done:
        target = reward
    else:
        target = reward + gamma * next_q[next_action]

    td_error = target - q_vals[action]
    q_vals[action] += alpha * td_error
    st.session_state.Q[state] = q_vals


# TD(0) update for value function
def td0_update(state, reward, next_state, done, alpha, gamma):
    v = get_v(state)
    next_v = get_v(next_state)

    if done:
        target = reward
    else:
        target = reward + gamma * next_v

    td_error = target - v
    st.session_state.V[state] = v + alpha * td_error


# n-step TD update
def nstep_td_update(states, rewards, n, alpha, gamma):
    T = len(rewards)

    for t in range(T):
        # calculate n-step return
        G = 0
        for i in range(t, min(t + n, T)):
            G += (gamma ** (i - t)) * rewards[i]

        # add bootstrap value if not at end
        if t + n < T:
            G += (gamma ** n) * get_v(states[t + n])

        # update
        v = get_v(states[t])
        st.session_state.V[states[t]] = v + alpha * (G - v)


# Monte Carlo update
def monte_carlo_update(episode_data, gamma):
    # episode_data is list of (state, action, reward)
    G = 0
    visited = set()

    # go backwards through episode
    for t in range(len(episode_data) - 1, -1, -1):
        state, action, reward = episode_data[t]
        G = gamma * G + reward

        # first visit MC
        if (state, action) not in visited:
            visited.add((state, action))
            q_vals = get_q(state, 4)  # assume 4 actions

            key = (state, action)
            if 'mc_counts' not in st.session_state:
                st.session_state.mc_counts = {}
            if 'mc_sums' not in st.session_state:
                st.session_state.mc_sums = {}

            if key not in st.session_state.mc_counts:
                st.session_state.mc_counts[key] = 0
                st.session_state.mc_sums[key] = 0

            st.session_state.mc_counts[key] += 1
            st.session_state.mc_sums[key] += G

            q_vals[action] = st.session_state.mc_sums[key] / st.session_state.mc_counts[key]
            st.session_state.Q[state] = q_vals


# one episode with Q-learning
def run_qlearning_episode(env, n_actions, alpha, gamma, epsilon, show_viz, env_placeholder, delay, env_name):
    state, _ = env.reset()
    total_reward = 0
    steps = 0

    for _ in range(500):
        action = choose_action(state, n_actions, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        qlearning_update(state, action, reward, next_state, done, n_actions, alpha, gamma)

        total_reward += reward
        steps += 1

        if show_viz:
            img = render_env_with_qvalues(env, env_name)
            if img:
                env_placeholder.image(img, use_container_width=True)
            else:
                try:
                    frame = env.render()
                    if frame is not None:
                        env_placeholder.image(Image.fromarray(frame), use_container_width=True)
                except:
                    pass
            time.sleep(delay)

        state = next_state
        if done:
            break

    return total_reward, steps


# one episode with SARSA
def run_sarsa_episode(env, n_actions, alpha, gamma, epsilon, show_viz, env_placeholder, delay, env_name):
    state, _ = env.reset()
    action = choose_action(state, n_actions, epsilon)
    total_reward = 0
    steps = 0

    for _ in range(500):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_action = choose_action(next_state, n_actions, epsilon)

        sarsa_update(state, action, reward, next_state, next_action, done, n_actions, alpha, gamma)

        total_reward += reward
        steps += 1

        if show_viz:
            img = render_env_with_qvalues(env, env_name)
            if img:
                env_placeholder.image(img, use_container_width=True)
            else:
                try:
                    frame = env.render()
                    if frame is not None:
                        env_placeholder.image(Image.fromarray(frame), use_container_width=True)
                except:
                    pass
            time.sleep(delay)

        state = next_state
        action = next_action
        if done:
            break

    return total_reward, steps


# one episode with TD(0)
def run_td0_episode(env, n_actions, alpha, gamma, epsilon, show_viz, env_placeholder, delay, env_name):
    state, _ = env.reset()
    total_reward = 0
    steps = 0

    for _ in range(500):
        action = choose_action(state, n_actions, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        td0_update(state, reward, next_state, done, alpha, gamma)

        total_reward += reward
        steps += 1

        if show_viz:
            img = render_env_with_qvalues(env, env_name)
            if img:
                env_placeholder.image(img, use_container_width=True)
            else:
                try:
                    frame = env.render()
                    if frame is not None:
                        env_placeholder.image(Image.fromarray(frame), use_container_width=True)
                except:
                    pass
            time.sleep(delay)

        state = next_state
        if done:
            break

    return total_reward, steps


# one episode with n-step TD
def run_nstep_td_episode(env, n_actions, n, alpha, gamma, epsilon, show_viz, env_placeholder, delay, env_name):
    state, _ = env.reset()
    states = [state]
    rewards = []
    total_reward = 0

    done = False
    while not done:
        action = choose_action(state, n_actions, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(next_state)
        rewards.append(reward)
        total_reward += reward

        if show_viz:
            img = render_env_with_qvalues(env, env_name)
            if img:
                env_placeholder.image(img, use_container_width=True)
            else:
                try:
                    frame = env.render()
                    if frame is not None:
                        env_placeholder.image(Image.fromarray(frame), use_container_width=True)
                except:
                    pass
            time.sleep(delay)

        state = next_state

        if len(rewards) > 500:
            break

    # update values
    nstep_td_update(states, rewards, n, alpha, gamma)

    return total_reward, len(rewards)


# one episode with Monte Carlo
def run_mc_episode(env, n_actions, gamma, epsilon, show_viz, env_placeholder, delay, env_name):
    state, _ = env.reset()
    episode_data = []
    total_reward = 0

    done = False
    while not done:
        action = choose_action(state, n_actions, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_data.append((state, action, reward))
        total_reward += reward

        if show_viz:
            img = render_env_with_qvalues(env, env_name)
            if img:
                env_placeholder.image(img, use_container_width=True)
            else:
                try:
                    frame = env.render()
                    if frame is not None:
                        env_placeholder.image(Image.fromarray(frame), use_container_width=True)
                except:
                    pass
            time.sleep(delay)

        state = next_state

        if len(episode_data) > 500:
            break

    # update Q values
    monte_carlo_update(episode_data, gamma)

    return total_reward, len(episode_data)


# one episode with n-step SARSA
def run_nstep_sarsa_episode(env, n_actions, n, alpha, gamma, epsilon, show_viz, env_placeholder, delay, env_name):
    state, _ = env.reset()
    action = choose_action(state, n_actions, epsilon)

    states = [state]
    actions = [action]
    rewards = [0.0]  # R_0 not used
    total_reward = 0

    T = float('inf')
    t = 0

    while True:
        if t < T:
            next_state, reward, terminated, truncated, _ = env.step(actions[t])
            states.append(next_state)
            rewards.append(reward)
            total_reward += reward

            if show_viz:
                img = render_env_with_qvalues(env, env_name)
                if img:
                    env_placeholder.image(img, use_container_width=True)
                else:
                    try:
                        frame = env.render()
                        if frame is not None:
                            env_placeholder.image(Image.fromarray(frame), use_container_width=True)
                    except:
                        pass
                time.sleep(delay)

            if terminated or truncated:
                T = t + 1
            else:
                next_action = choose_action(next_state, n_actions, epsilon)
                actions.append(next_action)

        # update time
        tau = t - n + 1

        if tau >= 0:
            # calculate n-step return
            G = sum((gamma ** (i - tau - 1)) * rewards[i]
                    for i in range(tau + 1, min(tau + n, int(T)) + 1))

            if tau + n < T:
                s_key = states[tau + n]
                a_key = actions[tau + n]
                q_vals = get_q(s_key, n_actions)
                G += (gamma ** n) * q_vals[a_key]

            # update Q
            s = states[tau]
            a = actions[tau]
            q_vals = get_q(s, n_actions)
            q_vals[a] += alpha * (G - q_vals[a])
            st.session_state.Q[s] = q_vals

        if tau == T - 1:
            break
        t += 1

        if t > 500:
            break

    return total_reward, int(min(T, 500))


# plot training progress
def plot_progress(rewards, lengths):
    if len(rewards) < 2:
        return None

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Rewards", "Episode Length"),
                        row_heights=[0.5, 0.5], vertical_spacing=0.2)

    fig.add_trace(go.Scatter(y=rewards, mode='lines', name='Reward'), row=1, col=1)

    # moving average
    if len(rewards) >= 10:
        ma = pd.Series(rewards).rolling(10).mean()
        fig.add_trace(go.Scatter(y=ma, mode='lines', name='Avg (10)'), row=1, col=1)

    fig.add_trace(go.Scatter(y=lengths, mode='lines', name='Steps'), row=2, col=1)

    # wider y-axis for steps
    if lengths:
        max_len = max(lengths) if lengths else 100
        fig.update_yaxes(range=[0, max_len * 1.2], row=2, col=1)

    fig.update_layout(height=450)
    return fig


# plot value heatmap for grid envs
def plot_values(env_name):
    if "FrozenLake" in env_name:
        size = 8
        rows, cols = size, size
    elif "CliffWalking" in env_name:
        rows, cols = 4, 12
    else:
        return None

    values = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            # get max Q value or V value
            if state in st.session_state.Q:
                values[r, c] = np.max(st.session_state.Q[state])
            elif state in st.session_state.V:
                values[r, c] = st.session_state.V[state]

    fig = go.Figure(data=go.Heatmap(
        z=values,
        colorscale='RdBu',
        text=np.round(values, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(title="State Values", height=300, yaxis=dict(autorange='reversed'))
    return fig


# render environment with Q-values on each cell
def render_env_with_qvalues(env, env_name):
    from PIL import ImageDraw, ImageFont

    if "FrozenLake" in env_name:
        rows, cols = 8, 8
    elif "CliffWalking" in env_name:
        rows, cols = 4, 12
    else:
        return None

    # get env frame
    try:
        frame = env.render()
        if frame is None:
            return None
        img = Image.fromarray(frame)
    except:
        return None

    draw = ImageDraw.Draw(img)

    # calculate cell size
    img_w, img_h = img.size
    cell_w = img_w / cols
    cell_h = img_h / rows

    # try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 10)
        small_font = ImageFont.truetype("arial.ttf", 8)
    except:
        font = ImageFont.load_default()
        small_font = font

    # draw Q-values on each cell
    # actions: 0=Left, 1=Down, 2=Right, 3=Up
    for r in range(rows):
        for c in range(cols):
            state = r * cols + c
            x = c * cell_w
            y = r * cell_h

            if state in st.session_state.Q:
                q = st.session_state.Q[state]
                best_action = np.argmax(q)

                # colors: red for best action, black for others
                colors = ["black", "black", "black", "black"]
                colors[best_action] = "red"

                # Up (action 3) - top center
                draw.text((x + cell_w/2 - 8, y + 3), f"{q[3]:.2f}", fill=colors[3], font=small_font)
                # Down (action 1) - bottom center
                draw.text((x + cell_w/2 - 8, y + cell_h - 14), f"{q[1]:.2f}", fill=colors[1], font=small_font)
                # Left (action 0) - left center
                draw.text((x + 2, y + cell_h/2 - 5), f"{q[0]:.2f}", fill=colors[0], font=small_font)
                # Right (action 2) - right center
                draw.text((x + cell_w - 28, y + cell_h/2 - 5), f"{q[2]:.2f}", fill=colors[2], font=small_font)

            elif state in st.session_state.V:
                v = st.session_state.V[state]
                # show V value in center
                color = "blue" if v > 0 else "red" if v < 0 else "black"
                draw.text((x + cell_w/2 - 12, y + cell_h/2 - 6), f"{v:.2f}", fill=color, font=font)

    return img


# === DQN for Atari ===

# CNN for DQN
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# preprocess atari frame
def preprocess_frame(frame):
    # convert to grayscale
    img = Image.fromarray(frame)
    img = img.convert('L')
    # resize to 84x84
    img = img.resize((84, 84))
    # normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# stack 4 frames together
def stack_frames(frames):
    return np.stack(frames, axis=0)


# replay buffer
class ReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# one episode with DQN
def run_dqn_episode(env, model, target_model, optimizer, replay_buffer,
                   epsilon, gamma, batch_size, device, show_viz, env_placeholder, delay):
    # reset environment
    frame, _ = env.reset()
    processed = preprocess_frame(frame)
    frames = deque([processed] * 4, maxlen=4)
    state = stack_frames(frames)

    total_reward = 0
    steps = 0

    for i in range(10000):
        # select action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = model(state_tensor)
            action = q_vals.argmax().item()

        # take step
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # process next frame
        next_processed = preprocess_frame(next_frame)
        frames.append(next_processed)
        next_state = stack_frames(frames)

        # store in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        total_reward += reward
        steps += 1

        # train if enough samples
        if len(replay_buffer) >= batch_size:
            # sample batch
            b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)

            b_states = torch.FloatTensor(np.array(b_states)).to(device)
            b_actions = torch.LongTensor(b_actions).to(device)
            b_rewards = torch.FloatTensor(b_rewards).to(device)
            b_next_states = torch.FloatTensor(np.array(b_next_states)).to(device)
            b_dones = torch.FloatTensor(b_dones).to(device)

            # zero gradients BEFORE forward pass
            optimizer.zero_grad()

            # current Q values
            current_q = model(b_states).gather(1, b_actions.unsqueeze(1)).squeeze(1)

            # target Q values
            with torch.no_grad():
                next_q = target_model(b_next_states).max(1)[0]
                target_q = b_rewards + gamma * next_q * (1 - b_dones)

            # loss and update
            loss = nn.MSELoss()(current_q, target_q)
            loss.backward()
            optimizer.step()

        # show visualization
        if show_viz and steps % 4 == 0:
            try:
                img = env.render()
                if img is not None:
                    env_placeholder.image(Image.fromarray(img), use_container_width=True)
            except:
                pass
            time.sleep(delay)

        state = next_state
        if done:
            break

    return total_reward, steps


# === SIDEBAR ===
st.sidebar.title("Setup")

# environment selection
env_options = {
    "FrozenLake": ("FrozenLake-v1", {"map_name": "8x8", "is_slippery": True}),
    "CliffWalking": ("CliffWalking-v1", {}),
    "Taxi": ("Taxi-v3", {}),
    "CartPole": ("CartPole-v1", {}),
    "MountainCar": ("MountainCar-v0", {}),
    "Pong": ("ALE/Pong-v5", {}),
    "Breakout": ("ALE/Breakout-v5", {}),
}
env_name = st.sidebar.selectbox("Environment", list(env_options.keys()))

# check if atari
is_atari = env_name in ["Pong", "Breakout"]

# algorithm selection
if is_atari:
    algorithms = ["DQN"]
else:
    algorithms = ["Q-Learning", "SARSA", "TD(0)", "n-step TD", "n-step SARSA", "Monte Carlo"]
algo = st.sidebar.selectbox("Algorithm", algorithms)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")

gamma = st.sidebar.slider("Gamma (discount)", 0.0, 1.0, 0.99, 0.01)

if not is_atari:
    alpha = st.sidebar.slider("Alpha (learning rate)", 0.01, 1.0, 0.1, 0.01)
else:
    alpha = 0.0001  # fixed for DQN

epsilon = st.sidebar.slider("Epsilon (exploration)", 0.0, 1.0, 0.1, 0.01)

# n for n-step methods
n_step = 4
if algo in ["n-step TD", "n-step SARSA"]:
    n_step = st.sidebar.slider("n (steps)", 1, 20, 4)
    st.sidebar.info(f"n={n_step}: Higher n = less bias, more variance")

# DQN specific
batch_size = 32
if algo == "DQN":
    batch_size = st.sidebar.slider("Batch size", 16, 128, 32, 16)

st.sidebar.markdown("---")
num_episodes = st.sidebar.number_input("Episodes", 1, 20000, 100)
speed = st.sidebar.slider("Speed", 1, 100, 50)
show_live = st.sidebar.checkbox("Show live training", True)
if st.session_state.episode > 0:
    st.sidebar.markdown("---")
    st.sidebar.info(f"Trained: {st.session_state.episode} episodes")


# === MAIN ===
st.title("RL Algorithms Visualization")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Environment")
    env_placeholder = st.empty()
    status = st.empty()

    btn1, btn2, btn3, btn4 = st.columns(4)
    train_btn = btn1.button("Train", type="primary")
    pause_btn = btn2.button("Pause" if not st.session_state.paused else "Resume")
    episode_btn = btn3.button("1 Episode")
    reset_btn = btn4.button("Reset")

with col2:
    st.subheader("Progress")
    chart_placeholder = st.empty()
    st.subheader("Values")
    value_placeholder = st.empty()


# create environment
def make_env():
    env_id, kwargs = env_options[env_name]
    return gym.make(env_id, render_mode="rgb_array", **kwargs)


# discretize state for continuous envs
def discretize_state(state, env_name):
    if env_name == "CartPole":
        bins = [
            np.linspace(-2.4, 2.4, 10),
            np.linspace(-3, 3, 10),
            np.linspace(-0.2, 0.2, 10),
            np.linspace(-3, 3, 10),
        ]
        discrete = []
        for i, val in enumerate(state):
            discrete.append(int(np.digitize(val, bins[i])))
        return tuple(discrete)
    elif env_name == "MountainCar":
        bins = [
            np.linspace(-1.2, 0.6, 20),
            np.linspace(-0.07, 0.07, 20),
        ]
        discrete = []
        for i, val in enumerate(state):
            discrete.append(int(np.digitize(val, bins[i])))
        return tuple(discrete)
    else:
        return state


# run training
def train(num_eps):
    env = make_env()
    n_actions = env.action_space.n
    delay = max(0.001, (100 - speed) / 500)

    progress = st.progress(0)
    if algo == "DQN":
        if st.session_state.dqn_model is None:
            st.session_state.dqn_model = DQN(n_actions).to(device)
            st.session_state.dqn_target = DQN(n_actions).to(device)
            st.session_state.dqn_target.load_state_dict(st.session_state.dqn_model.state_dict())
            st.session_state.dqn_optimizer = optim.Adam(st.session_state.dqn_model.parameters(), lr=0.0001)
            st.session_state.replay_buffer = ReplayBuffer(10000)

    st.session_state.training = True
    st.session_state.paused = False

    for ep in range(num_eps):
        # check if paused
        if st.session_state.paused:
            status.warning(f"Paused at episode {st.session_state.episode}. Change params and click Train to continue.")
            break

        if algo == "Q-Learning":
            reward, steps = run_qlearning_episode(env, n_actions, alpha, gamma, epsilon, show_live, env_placeholder, delay, env_name)
        elif algo == "SARSA":
            reward, steps = run_sarsa_episode(env, n_actions, alpha, gamma, epsilon, show_live, env_placeholder, delay, env_name)
        elif algo == "TD(0)":
            reward, steps = run_td0_episode(env, n_actions, alpha, gamma, epsilon, show_live, env_placeholder, delay, env_name)
        elif algo == "n-step TD":
            reward, steps = run_nstep_td_episode(env, n_actions, n_step, alpha, gamma, epsilon, show_live, env_placeholder, delay, env_name)
        elif algo == "n-step SARSA":
            reward, steps = run_nstep_sarsa_episode(env, n_actions, n_step, alpha, gamma, epsilon, show_live, env_placeholder, delay, env_name)
        elif algo == "DQN":
            reward, steps = run_dqn_episode(
                env, st.session_state.dqn_model, st.session_state.dqn_target,
                st.session_state.dqn_optimizer, st.session_state.replay_buffer,
                epsilon, gamma, batch_size, device, show_live, env_placeholder, delay
            )
            if (st.session_state.episode + 1) % 10 == 0:
                st.session_state.dqn_target.load_state_dict(st.session_state.dqn_model.state_dict())
        else:  # Monte Carlo
            reward, steps = run_mc_episode(env, n_actions, gamma, epsilon, show_live, env_placeholder, delay, env_name)

        st.session_state.rewards.append(reward)
        st.session_state.lengths.append(steps)
        st.session_state.episode += 1

        progress.progress((ep + 1) / num_eps)
        status.info(f"Episode {st.session_state.episode}: Reward={reward:.1f}, Steps={steps}")

        fig = plot_progress(st.session_state.rewards, st.session_state.lengths)
        if fig:
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{ep}")

        val_fig = plot_values(env_name)
        if val_fig:
            value_placeholder.plotly_chart(val_fig, use_container_width=True, key=f"val_{ep}")

    progress.empty()
    env.close()
    save_weights()
    st.session_state.training = False
    if not st.session_state.paused:
        status.success(f"Done! Trained {num_eps} episodes (saved)")


# button handlers
if pause_btn:
    st.session_state.paused = not st.session_state.paused
    if st.session_state.paused:
        st.rerun()

if train_btn:
    train(num_episodes)

if episode_btn:
    env = make_env()
    n_actions = env.action_space.n

    # initialize DQN if needed
    if algo == "DQN":
        if st.session_state.dqn_model is None:
            st.session_state.dqn_model = DQN(n_actions).to(device)
            st.session_state.dqn_target = DQN(n_actions).to(device)
            st.session_state.dqn_target.load_state_dict(st.session_state.dqn_model.state_dict())
            st.session_state.dqn_optimizer = optim.Adam(st.session_state.dqn_model.parameters(), lr=0.0001)
            st.session_state.replay_buffer = ReplayBuffer(10000)

    if algo == "Q-Learning":
        reward, steps = run_qlearning_episode(env, n_actions, alpha, gamma, epsilon, True, env_placeholder, 0.05, env_name)
    elif algo == "SARSA":
        reward, steps = run_sarsa_episode(env, n_actions, alpha, gamma, epsilon, True, env_placeholder, 0.05, env_name)
    elif algo == "TD(0)":
        reward, steps = run_td0_episode(env, n_actions, alpha, gamma, epsilon, True, env_placeholder, 0.05, env_name)
    elif algo == "n-step TD":
        reward, steps = run_nstep_td_episode(env, n_actions, n_step, alpha, gamma, epsilon, True, env_placeholder, 0.05, env_name)
    elif algo == "n-step SARSA":
        reward, steps = run_nstep_sarsa_episode(env, n_actions, n_step, alpha, gamma, epsilon, True, env_placeholder, 0.05, env_name)
    elif algo == "DQN":
        reward, steps = run_dqn_episode(
            env, st.session_state.dqn_model, st.session_state.dqn_target,
            st.session_state.dqn_optimizer, st.session_state.replay_buffer,
            epsilon, gamma, batch_size, device, True, env_placeholder, 0.02
        )
    else:
        reward, steps = run_mc_episode(env, n_actions, gamma, epsilon, True, env_placeholder, 0.05, env_name)

    st.session_state.rewards.append(reward)
    st.session_state.lengths.append(steps)
    st.session_state.episode += 1

    status.info(f"Episode {st.session_state.episode}: Reward={reward:.1f}, Steps={steps}")

    fig = plot_progress(st.session_state.rewards, st.session_state.lengths)
    if fig:
        chart_placeholder.plotly_chart(fig, use_container_width=True, key="step_chart")

    env.close()
    save_weights()

if reset_btn:
    st.session_state.Q = {}
    st.session_state.V = {}
    st.session_state.rewards = []
    st.session_state.lengths = []
    st.session_state.episode = 0
    if 'mc_counts' in st.session_state:
        del st.session_state.mc_counts
    if 'mc_sums' in st.session_state:
        del st.session_state.mc_sums
    # reset DQN
    st.session_state.dqn_model = None
    st.session_state.dqn_target = None
    st.session_state.dqn_optimizer = None
    st.session_state.replay_buffer = None
    # delete saved files
    delete_weights()
    st.rerun()


# show initial env
if st.session_state.episode == 0:
    try:
        env = make_env()
        env.reset()
        frame = env.render()
        if frame is not None:
            env_placeholder.image(Image.fromarray(frame), use_container_width=True)
        env.close()
    except Exception as e:
        env_placeholder.write("Click Train to start")

# show existing data
if st.session_state.rewards:
    fig = plot_progress(st.session_state.rewards, st.session_state.lengths)
    if fig:
        chart_placeholder.plotly_chart(fig, use_container_width=True, key="init_chart")

    val_fig = plot_values(env_name)
    if val_fig:
        value_placeholder.plotly_chart(val_fig, use_container_width=True, key="init_val")
