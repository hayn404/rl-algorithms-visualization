# RL Algorithms Visualization Tool

## Algorithms Implemented

### Tabular Methods
| Algorithm | Description |
|-----------|-------------|
| Q-Learning | Off-policy TD control, updates using max Q(s',a') |
| SARSA | On-policy TD control, updates using Q(s',a') from policy |
| TD(0) | Temporal difference for value function estimation |
| n-step TD | Multi-step TD with configurable n |
| n-step SARSA | Multi-step on-policy control |
| Monte Carlo | First-visit MC, updates after episode completion |

### Deep RL
| Algorithm | Description |
|-----------|-------------|
| DQN | Deep Q-Network with CNN, experience replay, target network |

## Environments

| Environment | Type | State Space | Actions |
|-------------|------|-------------|---------|
| FrozenLake (8x8) | Grid | 64 discrete states | 4 (Up, Down, Left, Right) |
| CliffWalking | Grid | 48 discrete states | 4 (Up, Down, Left, Right) |
| Taxi | Grid | 500 discrete states | 6 (Move + Pickup/Dropoff) |
| CartPole | Classic Control | 4 continuous | 2 (Left, Right) |
| MountainCar | Classic Control | 2 continuous | 3 (Left, None, Right) |
| Pong | Atari | 210x160 RGB | 6 |
| Breakout | Atari | 210x160 RGB | 4 |

## Parameter Adjustment

| Parameter | Range | Effect |
|-----------|-------|--------|
| Gamma (γ) | 0.0 - 1.0 | Discount factor for future rewards |
| Alpha (α) | 0.01 - 1.0 | Learning rate |
| Epsilon (ε) | 0.0 - 1.0 | Exploration rate |
| n (n-step) | 1 - 20 | Steps for n-step methods |
| Batch Size | 16 - 128 | DQN training batch size |
| Episodes | 1 - 20000 | Training duration |
| Speed | 1 - 100 | Visualization speed |

## Visualization Techniques

### Training Progress
- Real-time reward plot per episode
- Moving average (10 episodes) for trend analysis
- Episode length tracking

### Environment Visualization
- Live rendering during training
- Q-values overlay on grid cells (FrozenLake, CliffWalking)
  - Shows action values for all 4 directions
  - Best action highlighted in red
  - V-values shown in center (blue=positive, red=negative)

### Value Function
- Heatmap of state values for grid environments
- Color-coded by value magnitude

## Technical Implementation

- **Framework**: Streamlit web interface
- **RL Library**: Gymnasium
- **Deep Learning**: PyTorch
- **Visualization**: Plotly, PIL
- **Cloud Training**: Modal (GPU support for DQN)

## File Structure

```
python-rl/
├── app.py           # Main application
├── modal_train.py   # Cloud GPU training
├── requirements.txt # Dependencies
└── saved_weights/   # Auto-saved model weights
```
