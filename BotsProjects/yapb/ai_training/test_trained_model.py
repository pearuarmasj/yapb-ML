import numpy as np
import torch
import torch.nn as nn
import cv2
import time
import mss
import pydirectinput
import ctypes
import os
import glob
import re
from collections import deque
from datetime import datetime

def move_mouse_relative(x, y):
    ctypes.windll.user32.mouse_event(0x0001, x, y, 0, 0)

def grab_screen(region=None):
    with mss.mss() as sct:
        monitor = sct.monitors[1] if not region else region
        img = np.array(sct.grab(monitor))
        return img[..., :3]

def preprocess(frame, size=(84, 84)):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size)
    return frame.astype(np.uint8)

class CNNPolicy(nn.Module):
    def __init__(self, num_actions=12):
        super(CNNPolicy, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out_size((4, 84, 84))
        
        self.value_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )
        
    def _get_conv_out_size(self, shape):
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.size()))
    
    def forward(self, x):
        x = x.float() / 255.0
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        value = self.value_head(conv_out)
        policy = self.policy_head(conv_out)
        
        return policy, value

class TrainedBot:
    def __init__(self, model_path="assaultcube_offline_trained.pth", region=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNPolicy().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.region = region
        self.frame_stack = deque(maxlen=4)
        self.held_keys = set()
        
        self.actions = [
            ('w',), ('a',), ('s',), ('d',),
            ('w','mouse_left'), ('w','mouse_right'),
            ('a','mouse_left'), ('a','mouse_right'),
            ('s','mouse_left'), ('s','mouse_right'),
            ('d','mouse_left'), ('d','mouse_right')
        ]
        
        print(f"Loaded model on {self.device}")
    
    def reset_keys(self):
        for key in list(self.held_keys):
            pydirectinput.keyUp(key)
        self.held_keys.clear()
    
    def execute_action(self, action_idx):
        action = self.actions[action_idx]
        
        used_wasd = False
        used_mouse = False
        
        if isinstance(action, tuple) and len(action) >= 1:
            key = action[0]
            mouse = action[1] if len(action) > 1 else None
        else:
            key = action
            mouse = None
            
        current_wasd = key if key in ['w', 'a', 's', 'd'] else None
        
        for held_key in list(self.held_keys):
            if held_key != current_wasd:
                pydirectinput.keyUp(held_key)
                self.held_keys.remove(held_key)
        
        if current_wasd and current_wasd not in self.held_keys:
            pydirectinput.keyDown(current_wasd)
            self.held_keys.add(current_wasd)
            used_wasd = True
        elif current_wasd:
            used_wasd = True
            
        if mouse == 'mouse_left':
            move_mouse_relative(-50, 0)
            used_mouse = True
        elif mouse == 'mouse_right':
            move_mouse_relative(50, 0)
            used_mouse = True
        
        return used_wasd, used_mouse
    
    def get_action(self, observation):
        if len(self.frame_stack) < 4:
            for _ in range(4):
                self.frame_stack.append(observation)
        else:
            self.frame_stack.append(observation)
        
        stacked_obs = np.stack(self.frame_stack, axis=0)
        obs_tensor = torch.tensor(stacked_obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_probs, value = self.model(obs_tensor)
            action_idx = torch.multinomial(policy_probs, 1).item()
        
        return action_idx, value.item()
    
    def run(self, duration_seconds=300):
        print("Starting trained bot...")
        print("Make sure AssaultCube is running and visible!")
        time.sleep(3)
        
        start_time = time.time()
        step_count = 0
        total_value = 0
        
        while time.time() - start_time < duration_seconds:
            obs = grab_screen(self.region)
            processed_obs = preprocess(obs)
            
            action_idx, value = self.get_action(processed_obs)
            used_wasd, used_mouse = self.execute_action(action_idx)
            
            step_count += 1
            total_value += value
            
            if step_count % 100 == 0:
                avg_value = total_value / step_count
                elapsed = time.time() - start_time
                print(f"Step {step_count} | Action: {action_idx} | Value: {value:.4f} | Avg Value: {avg_value:.4f} | Time: {elapsed:.1f}s")
            
            time.sleep(0.1)
        
        self.reset_keys()
        print(f"Bot completed {step_count} steps in {duration_seconds} seconds")
        print(f"Average value: {total_value / step_count:.4f}")

import glob
import re
from datetime import datetime

def find_available_models():
    """Find all trained model files and extract their details"""
    model_files = glob.glob("assaultcube_offline_trained_*.pth")
    models = []
    
    for file_path in model_files:
        filename = os.path.basename(file_path)
        
        # Extract details from filename
        match = re.search(r'batch(\d+)_epochs(\d+)_(\d{8}_\d{6})', filename)
        if match:
            batch_size = int(match.group(1))
            epochs = int(match.group(2))
            timestamp_str = match.group(3)
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                # Get file size
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                models.append({
                    'path': file_path,
                    'filename': filename,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'timestamp': timestamp,
                    'size_mb': file_size
                })
            except ValueError:
                continue
    
    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return models

def select_model():
    """Display available models and let user select one"""
    models = find_available_models()
    
    if not models:
        print("No trained models found!")
        print("Train a model first using offline_trainer.py")
        return None
    
    print("Available trained models:")
    print("=" * 80)
    
    for i, model in enumerate(models):
        print(f"{i+1:2d}. {model['filename']}")
        print(f"    Batch Size: {model['batch_size']}")
        print(f"    Epochs: {model['epochs']}")
        print(f"    Trained: {model['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Size: {model['size_mb']:.1f} MB")
        print()
    
    while True:
        try:
            choice = input(f"Select model (1-{len(models)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"Selected: {selected['filename']}")
                return selected['path']
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def main():
    region = {"top": 0, "left": 0, "width": 2560, "height": 1440}
    
    model_path = select_model()
    if not model_path:
        print("No model selected. Exiting.")
        return
    
    try:
        bot = TrainedBot(model_path=model_path, region=region)
        bot.run(duration_seconds=300)
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found!")
        print("The file may have been moved or deleted.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
