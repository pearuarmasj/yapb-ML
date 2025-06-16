import mss
import numpy as np
import cv2
import time
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import random

print("Wait 3 seconds to switch to the game window...")

time.sleep(3)  # Allow time to switch to the game window

def grab_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if not region else region
        img = np.array(sct.grab(monitor))
        # Only keep RGB, drop alpha
        return img[..., :3]

# Example: Grab full screen
frame = grab_screen()
cv2.imshow('What I See', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Example: Move mouse, press key
pyautogui.moveTo(400, 400)   # Absolute coords
pyautogui.keyDown('w')       # Press W key (move forward)
time.sleep(1)
pyautogui.keyUp('w')
pyautogui.click(button='left')  # Shoot

class CS16Env:
    def __init__(self, region=None):
        self.region = region
        # Define possible actions: forward, left, right, back, shoot
        self.actions = ['w', 'a', 's', 'd', 'shoot']

    def reset(self):
        # Respawn logic if needed, or just start at initial position
        pass

    def step(self, action):
        if action in ['w', 'a', 's', 'd']:
            pyautogui.keyDown(action)
            time.sleep(0.1)
            pyautogui.keyUp(action)
        elif action == 'shoot':
            pyautogui.click(button='left')
        # Get new screen/frame
        obs = grab_screen(self.region)
        # Reward: For now, just random
        reward = random.random()
        done = False  # TODO: Detect if dead or episode ends
        return obs, reward, done, {}

    def get_observation(self):
        return grab_screen(self.region)

def preprocess(frame, size=(84, 84)):
    # Resize, convert to grayscale, normalize
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size)
    frame = frame.astype(np.float32) / 255.0  # Normalize
    return frame  # Shape: (84, 84)

from collections import deque

frame_stack = deque(maxlen=4)

def reset_stack(frame):
    frame_stack.clear()
    for _ in range(4):
        frame_stack.append(frame)

def stack_frames(new_frame):
    frame_stack.append(new_frame)
    return np.stack(frame_stack, axis=0)  # Shape: (4, 84, 84)

class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4),  # (4,84,84) -> (16,20,20)
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),           # (16,20,20) -> (32,9,9)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, exp):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

env = CS16Env(region=None)  # Define region if needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dqn = DQN().to(device)
optimizer = optim.Adam(dqn.parameters(), lr=1e-4)
buffer = ReplayBuffer(10000)
BATCH_SIZE = 32
GAMMA = 0.99

obs = preprocess(env.get_observation())
reset_stack(obs)
state = stack_frames(obs)

for step in range(10_000):  # Or however long you want to suffer
    # ε-greedy action selection
    if random.random() < 0.05:
        action_idx = random.randint(0, 4)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_vals = dqn(state_tensor)
            action_idx = q_vals.argmax().item()
    action = env.actions[action_idx]
    obs2, reward, done, info = env.step(action)
    obs2_proc = preprocess(obs2)
    next_state = stack_frames(obs2_proc)
    buffer.push((state, action_idx, reward, next_state, done))
    state = next_state

    # Learn every 4 steps, if buffer big enough
    if len(buffer.buffer) > BATCH_SIZE and step % 4 == 0:
        batch = buffer.sample(BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)
        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a).unsqueeze(1).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)

        q_vals = dqn(s).gather(1, a)
        with torch.no_grad():
            next_q = dqn(s2).max(1)[0].unsqueeze(1)
        target = r + GAMMA * next_q * (1 - d)
        loss = ((q_vals - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
