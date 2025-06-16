import mss
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pydirectinput
import matplotlib.pyplot as plt
from collections import deque
import ctypes
from ctypes import wintypes
import win32gui
import win32api
import win32process
import psutil

print("Wait 3 seconds to switch to the game window...")
time.sleep(3)

region = {"top": 0, "left": 0, "width": 2560, "height": 1440}

def move_mouse_relative(x, y):
    ctypes.windll.user32.mouse_event(0x0001, x, y, 0, 0)  # MOUSEEVENTF_MOVE

def get_window_info():
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            # Get process ID for this window
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                # Get process name
                process = psutil.Process(pid)
                process_name = process.name().lower()
                
                # Check if it's AssaultCube
                if "AssaultCube" in process_name or "ac_client.exe" in process_name:
                    rect = win32gui.GetWindowRect(hwnd)
                    x = rect[0]
                    y = rect[1]
                    w = rect[2] - x
                    h = rect[3] - y
                    window_text = win32gui.GetWindowText(hwnd)
                    print(f"Window: {window_text} (Process: {process_name})")
                    print(f"Position: x={x}, y={y}")
                    print(f"Size: {w}x{h}")
                    print(f"Region dict: {{'top': {y}, 'left': {x}, 'width': {w}, 'height': {h}}}")
                    windows.append(hwnd)
            except:
                pass
        return True
    
    windows = []
    win32gui.EnumWindows(callback, windows)
    
    if not windows:
        print("AssaultCube window not found. Make sure the game is running.")

get_window_info()

def grab_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if not region else region
        img = np.array(sct.grab(monitor))
        return img[..., :3]


class CS16Env:
    def __init__(self, region=None):
        self.region = region
        # Macro actions: always WASD + mouse combination
        self.actions = [
            ('w','mouse_left'), ('w','mouse_right'),
            ('a','mouse_left'), ('a','mouse_right'),
            ('s','mouse_left'), ('s','mouse_right'),
            ('d','mouse_left'), ('d','mouse_right')
        ]
        # Track which keys are currently held down
        self.held_keys = set()
        
    def reset(self):
        # Release all held keys on reset
        for key in list(self.held_keys):
            pydirectinput.keyUp(key)
        self.held_keys.clear()
        
    def step(self, action):
        used_wasd = False
        used_mouse = False
        
        # Parse action
        if isinstance(action, tuple) or isinstance(action, list):
            if len(action) == 2:
                key, mouse = action
            else:
                key = action[0]
                mouse = None
        else:
            key = action
            mouse = None
            
        # Release keys that aren't part of current action
        current_wasd = key if key in ['w', 'a', 's', 'd'] else None
        for held_key in list(self.held_keys):
            if held_key != current_wasd:
                pydirectinput.keyUp(held_key)
                self.held_keys.remove(held_key)
        
        # Hold down the current WASD key
        if current_wasd and current_wasd not in self.held_keys:
            pydirectinput.keyDown(current_wasd)
            self.held_keys.add(current_wasd)
            used_wasd = True
        elif current_wasd:
            used_wasd = True  # Key still being held
            
        # Handle mouse movement
        if mouse == 'mouse_left':
            move_mouse_relative(-50, 0)
            used_mouse = True
        elif mouse == 'mouse_right':
            move_mouse_relative(50, 0)
            used_mouse = True
            
        obs = grab_screen(self.region)
        reward = random.random()  # gets overwritten
        done = False
        return obs, reward, done, {'used_wasd': used_wasd, 'used_mouse': used_mouse}
        
    def get_observation(self):
        return grab_screen(self.region)

def preprocess(frame, size=(84, 84)):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size)
    frame = frame.astype(np.float32) / 255.0
    return frame

frame_stack = deque(maxlen=4)
def reset_stack(frame):
    frame_stack.clear()
    for _ in range(4):
        frame_stack.append(frame)
def stack_frames(new_frame):
    frame_stack.append(new_frame)
    return np.stack(frame_stack, axis=0)

class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
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
        recent = self.buffer[-50:] if len(self.buffer) >= 50 else self.buffer
        n_recent = batch_size // 2
        n_random = batch_size - n_recent
        batch = []
        if recent:
            batch.extend(random.choices(recent, k=n_recent))
        if len(self.buffer) > 0:
            batch.extend(random.choices(self.buffer, k=n_random))
        return batch

env = CS16Env(region=region)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dqn = DQN(in_channels=4, n_actions=len(env.actions)).to(device)
target_dqn = DQN(in_channels=4, n_actions=len(env.actions)).to(device)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()
optimizer = optim.Adam(dqn.parameters(), lr=1e-4)
buffer = ReplayBuffer(10000)
BATCH_SIZE = 32
GAMMA = 0.99
rewards = []

obs = preprocess(env.get_observation())
reset_stack(obs)
state = stack_frames(obs)
prev_frame = obs

stuck_counter = 0
STUCK_LIMIT = 20
STUCK_THRESH = 1e-4
TARGET_UPDATE_FREQ = 1000

for step in range(10000):
    # ε-greedy over macro-actions
    if random.random() < 0.05:
        action_idx = random.randint(0, len(env.actions) - 1)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_vals = dqn(state_tensor)
            action_idx = q_vals.argmax().item()
    action = env.actions[action_idx]
    obs2, _, done, info = env.step(action)
    obs2_proc = preprocess(obs2)
    next_state = stack_frames(obs2_proc)
    diff = np.abs(obs2_proc.astype(np.float32) - prev_frame.astype(np.float32))
    reward = np.mean(diff)
    
    # Enhanced stuck detection
    if reward < STUCK_THRESH:
        stuck_counter += 1
        if stuck_counter > 5:
            reward = -0.5 - (stuck_counter * 0.1)
        else:
            reward = -0.2
    else:
        stuck_counter = 0
    prev_frame = obs2_proc

    # --- Novelty exploration reward with dual-control gating ---
    small_obs = cv2.resize(obs2_proc, (16, 16)).astype(np.uint8)
    novelty_hash = tuple(small_obs.flatten())
    if not hasattr(globals(), 'novelty_set'):
        novelty_set = set()
        novelty_history = deque(maxlen=1000)
        action_history = deque(maxlen=10)
        mouse_direction_history = deque(maxlen=8)
        globals()['novelty_set'] = novelty_set
        globals()['novelty_history'] = novelty_history
        globals()['action_history'] = action_history
        globals()['mouse_direction_history'] = mouse_direction_history
    else:
        novelty_set = globals()['novelty_set']
        novelty_history = globals()['novelty_history']
        action_history = globals()['action_history']
        mouse_direction_history = globals()['mouse_direction_history']
    used_wasd = info.get('used_wasd', False)
    used_mouse = info.get('used_mouse', False)
      # Track action patterns for anti-gaming
    action_history.append(action)
    if len(action) == 2 and action[1] == 'mouse_left':
        mouse_direction_history.append('left')
    elif len(action) == 2 and action[1] == 'mouse_right':
        mouse_direction_history.append('right')

    if novelty_hash not in novelty_set and used_wasd and used_mouse and reward > STUCK_THRESH:
        reward += 0.3
        novelty_set.add(novelty_hash)
        novelty_history.append(novelty_hash)
        if len(novelty_set) > 1200:
            to_remove = novelty_history.popleft()
            novelty_set.remove(to_remove)
    elif novelty_hash not in novelty_set:
        reward -= 0.2
      # Apply reward reduction instead of punishment for single input usage
    if used_wasd and used_mouse:
        pass
    elif used_wasd or used_mouse:
        reward *= 0.1
    else:
        reward *= 0.05
    
    # Anti-gaming mechanisms
    gaming_penalty = 0.0
    
    # Detect repetitive action patterns
    if len(action_history) >= 6:
        recent_actions = list(action_history)[-6:]
        if len(set(recent_actions)) <= 2:
            gaming_penalty += 0.4
    
    # Detect mouse direction bias or flicking
    if len(mouse_direction_history) >= 6:
        recent_mouse = list(mouse_direction_history)[-6:]
        left_count = recent_mouse.count('left')
        right_count = recent_mouse.count('right')
        
        # Punish extreme directional bias
        if left_count >= 5 or right_count >= 5:
            gaming_penalty += 0.3
        
        # Punish alternating flick patterns
        if len(recent_mouse) >= 4:
            is_alternating = all(
                recent_mouse[i] != recent_mouse[i+1] 
                for i in range(len(recent_mouse)-1)
            )
            if is_alternating:
                gaming_penalty += 0.5
    
    # Apply gaming penalties
    if gaming_penalty > 0:
        reward *= max(0.01, 1.0 - gaming_penalty)

    rewards.append(reward)
    if stuck_counter >= STUCK_LIMIT:
        print(f'Agent stuck at step {step}! Resetting episode...')
        done = True
        stuck_counter = 0

    print(f"Step: {step} | Action: {action} | Reward: {reward:.5f} | Done: {done}")
    if step % 10 == 0:
        cv2.imshow('Bot POV', obs2_proc)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted, quitting.")
            break
    buffer.push((state, action_idx, reward, next_state, done))
    state = next_state

    if len(buffer.buffer) > BATCH_SIZE:
        batch = buffer.sample(BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)
        s = torch.from_numpy(np.stack(s)).float().to(device)
        a = torch.tensor(a).unsqueeze(1).to(device)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
        s2 = torch.from_numpy(np.stack(s2)).float().to(device)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)
        q_vals = dqn(s).gather(1, a)
        with torch.no_grad():
            next_actions = dqn(s2).max(1)[1].unsqueeze(1)
            next_q = target_dqn(s2).gather(1, next_actions)
        target = r + GAMMA * next_q * (1 - d)
        loss = ((q_vals - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Train Step: {step} | Loss: {loss.item():.5f}")
        if device.type == 'cuda' and step % 50 == 0:
            print(f"Current VRAM usage: {torch.cuda.memory_allocated() // 1024**2} MB")
            print(f"Max VRAM allocated: {torch.cuda.max_memory_allocated() // 1024**2} MB")
        if step % TARGET_UPDATE_FREQ == 0:
            target_dqn.load_state_dict(dqn.state_dict())
            print("Updated target network.")
    if done:
        print(f"--- Episode finished at step {step}. Resetting... ---")
        obs = preprocess(env.get_observation())
        reset_stack(obs)
        state = stack_frames(obs)
        prev_frame = obs

cv2.destroyAllWindows()
env.reset()
plt.plot(rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Reward per Step')
plt.show()
