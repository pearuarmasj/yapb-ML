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
from stable_baselines3 import dqn
from stable_baselines3 import PPO

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
        self.region = region        # WASD-only actions (preferred) + WASD+mouse combinations (for when stuck)
        self.actions = [
            ('w',), ('a',), ('s',), ('d',),  # WASD-only actions
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
    def __init__(self, in_channels=4, n_actions=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 8, stride=4),  # 4x more channels
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),  # 4x more channels
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1),  # Extra conv layer
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*7*7, 1024),  # Much larger dense layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization
            nn.Linear(1024, 512),  # Extra dense layer
            nn.ReLU(),            nn.Linear(512, n_actions)
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
        # More sophisticated sampling for larger batches
        if len(self.buffer) < batch_size:
            return self.buffer
        
        recent = self.buffer[-200:] if len(self.buffer) >= 200 else self.buffer
        n_recent = min(batch_size // 3, len(recent))  # 1/3 recent experiences
        n_random = batch_size - n_recent
        
        batch = []
        if recent and n_recent > 0:
            batch.extend(random.choices(recent, k=n_recent))
        if len(self.buffer) > 0 and n_random > 0:
            batch.extend(random.choices(self.buffer, k=n_random))
        return batch

env = CS16Env(region=region)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dqn = DQN(in_channels=4, n_actions=len(env.actions)).to(device)
target_dqn = DQN(in_channels=4, n_actions=len(env.actions)).to(device)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()
optimizer = optim.Adam(dqn.parameters(), lr=3e-4)
buffer = ReplayBuffer(100000)  # 10x larger buffer
BATCH_SIZE = 256  # Massive batch size for your 4080
GAMMA = 0.99
rewards = []

obs = preprocess(env.get_observation())
reset_stack(obs)
state = stack_frames(obs)
prev_frame = obs

stuck_counter = 0
STUCK_LIMIT = 20
STUCK_THRESH = 1e-4
TARGET_UPDATE_FREQ = 200  # More frequent updates with bigger batches

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
    reward = np.mean(diff)    # Enhanced stuck detection
    if reward < STUCK_THRESH:
        stuck_counter += 1
        if stuck_counter > 5:
            reward = -0.5 - (stuck_counter * 0.1)
        else:
            reward = -0.2
    else:
        stuck_counter = 0
    prev_frame = obs2_proc# --- Enhanced exploration system ---
    # Fine-grained novelty (existing)
    small_obs = cv2.resize(obs2_proc, (16, 16)).astype(np.uint8)
    novelty_hash = tuple(small_obs.flatten())
      # Coarse exploration tracking - much coarser than before
    region_obs = cv2.resize(obs2_proc, (2, 2)).astype(np.uint8)
    region_hash = tuple(region_obs.flatten())
    
    # Ultra-coarse area tracking for camping detection
    area_obs = cv2.resize(obs2_proc, (1, 2)).astype(np.uint8)  # Just 2 pixels
    area_hash = tuple(area_obs.flatten())
    
    if not hasattr(globals(), 'novelty_set'):
        novelty_set = set()
        novelty_history = deque(maxlen=1000)
        action_history = deque(maxlen=10)
        mouse_direction_history = deque(maxlen=8)
        region_set = set()
        region_history = deque(maxlen=80)
        region_visit_count = {}
        area_history = deque(maxlen=30)
        globals()['novelty_set'] = novelty_set
        globals()['novelty_history'] = novelty_history
        globals()['action_history'] = action_history
        globals()['mouse_direction_history'] = mouse_direction_history
        globals()['region_set'] = region_set
        globals()['region_history'] = region_history
        globals()['region_visit_count'] = region_visit_count
        globals()['area_history'] = area_history
    else:
        novelty_set = globals()['novelty_set']
        novelty_history = globals()['novelty_history']
        action_history = globals()['action_history']
        mouse_direction_history = globals()['mouse_direction_history']
        region_set = globals()['region_set']
        region_history = globals()['region_history']
        region_visit_count = globals()['region_visit_count']
        area_history = globals()['area_history']
    used_wasd = info.get('used_wasd', False)
    used_mouse = info.get('used_mouse', False)    # Strategic mouse incentive - encourage mouse when stuck
    if stuck_counter > 2 and used_mouse:
        reward += 0.8  # Much bigger bonus for using mouse when stuck
      # Track action patterns for anti-gaming
    action_history.append(action)
    if len(action) == 2 and action[1] == 'mouse_left':
        mouse_direction_history.append('left')
    elif len(action) == 2 and action[1] == 'mouse_right':
        mouse_direction_history.append('right')
    
    # Action spam detection - punish repeated same actions
    if len(action_history) >= 30:
        recent_actions = list(action_history)[-30:]
        same_action_count = recent_actions.count(action)
        if same_action_count >= 15:  # Same action 15+ times in last 30 steps
            reward = -3.0  # Massive punishment for action spam

    if novelty_hash not in novelty_set and used_wasd and used_mouse and reward > STUCK_THRESH:
        reward += 0.3
        novelty_set.add(novelty_hash)
        novelty_history.append(novelty_hash)
        if len(novelty_set) > 1200:
            to_remove = novelty_history.popleft()
            novelty_set.remove(to_remove)
    elif novelty_hash not in novelty_set:
        reward -= 0.2    # Track region visits and penalize camping
    region_visit_count[region_hash] = region_visit_count.get(region_hash, 0) + 1
    region_history.append(region_hash)
    area_history.append(area_hash)
    
    # Movement stagnation detection - same region for multiple steps
    if len(region_history) >= 5:
        recent_regions_for_movement = list(region_history)[-5:]
        if len(set(recent_regions_for_movement)) == 1:  # Same region for 5 steps
            reward = -2.5  # Heavy punishment for not moving
      # Major exploration bonuses - but only for genuinely new areas
    if region_hash not in region_set:
        region_set.add(region_hash)
        reward += 1.0  # Big bonus for new areas
    
    # Ultra-harsh camping penalty using multiple resolution tracking
    recent_areas = list(area_history)[-15:]  # Shorter window, more aggressive
    if len(recent_areas) >= 10:  # Trigger sooner
        unique_areas = len(set(recent_areas))
        if unique_areas <= 1:
            reward -= 8.0  # Nuclear penalty for staying put
        elif unique_areas <= 2:
            reward -= 5.0  # Massive penalty for minimal movement
        elif unique_areas <= 3:
            reward -= 3.0  # Heavy penalty for very limited exploration
    
    # Tunnel running detection - more aggressive
    recent_regions = list(region_history)[-8:]  # Shorter window
    if len(recent_regions) >= 6:  # Trigger sooner
        unique_recent = len(set(recent_regions))
        if unique_recent <= 2:
            reward -= 4.0  # Massive penalty for back-and-forth
        elif unique_recent <= 3:
            reward -= 2.0  # Heavy penalty for limited area coverage
    
    # Progressive visit frequency punishment - more aggressive
    visit_count = region_visit_count[region_hash]
    if visit_count > 3:  # Trigger sooner
        visit_penalty = min(2.0, visit_count * 0.2)  # Much harsher scaling
        reward -= visit_penalty
    
    # Oscillation detection (back-and-forth movement) - more sensitive
    if len(region_history) >= 4:  # Shorter pattern detection
        last_4 = list(region_history)[-4:]
        if len(set(last_4)) <= 2 and len(last_4) == 4:
            reward -= 3.0  # Punish oscillation patterns harder
    
    # Fallback anti-camping: if reward is still positive after penalties, force negative
    if stuck_counter == 0 and reward > 0.5:  # If somehow still getting good rewards
        recent_unique = len(set(list(area_history)[-10:]))
        if recent_unique <= 2:
            reward = -2.0  # Force negative for camping# Apply reward adjustments with strong WASD-only preference
    # BUT respect stuck detection first
    if stuck_counter > 0:
        # When stuck, don't override the negative stuck rewards
        if used_wasd and used_mouse and stuck_counter > 3:
            # Only bonus for mouse when severely stuck
            reward += 0.2
    else:
        # When not stuck, apply normal WASD preference
        base_reward = max(0.01, reward)
        
        if used_wasd and used_mouse:
            reward = base_reward + 0.3  # Bonus for using both inputs
        elif used_wasd:
            reward = base_reward + 0.1  # Smaller bonus for WASD-only
        elif used_mouse:
            reward = base_reward * 0.1
        else:
            reward = base_reward * 0.05
    
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
                gaming_penalty += 0.5    # Punish consecutive mouse usage (anti-spam) - very lenient approach
    consecutive_mouse = 0
    for i in range(min(20, len(action_history))):
        action_check = list(action_history)[-(i+1)]
        if len(action_check) == 2:  # Has mouse component
            consecutive_mouse += 1
        else:
            break
    
    if consecutive_mouse >= 15:  # Allow up to 15 consecutive mouse actions
        gaming_penalty += 0.3  # Light penalty for extreme spam
    elif consecutive_mouse >= 20:  # Very heavy spam
        gaming_penalty += 0.6
    
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
