import subprocess
import os
import time
import threading
from datetime import datetime
import sys
sys.path.append('.')

# Import region selector from main script
try:
    from OpenAI_Gym_Experiments import RegionSelector
except ImportError:
    # Fallback region selector if import fails
    import tkinter as tk
    
    class FallbackRegionSelector:
        def __init__(self):
            self.start_x = 0
            self.start_y = 0
            self.end_x = 0
            self.end_y = 0
            self.rect = None
            self.root = None
            self.canvas = None
            self.selected_region = None
            
        def select_region(self):
            self.root = tk.Tk()
            self.root.attributes('-fullscreen', True)
            self.root.attributes('-alpha', 0.3)
            self.root.attributes('-topmost', True)
            self.root.configure(bg='black')
            
            self.canvas = tk.Canvas(self.root, highlightthickness=0)
            self.canvas.pack(fill=tk.BOTH, expand=True)
            
            self.canvas.bind('<Button-1>', self.on_click)
            self.canvas.bind('<B1-Motion>', self.on_drag)
            self.canvas.bind('<ButtonRelease-1>', self.on_release)
            
            self.root.bind('<Escape>', lambda e: self.cancel_selection())
            
            self.canvas.create_text(
                self.root.winfo_screenwidth() // 2, 50,
                text="Drag to select region. Press ESC to cancel.",
                fill='white', font=('Arial', 16)
            )
            
            self.root.mainloop()
            return self.selected_region
            
        def on_click(self, event):
            self.start_x = event.x
            self.start_y = event.y
            
        def on_drag(self, event):
            if self.rect and self.canvas:
                self.canvas.delete(self.rect)
            if self.canvas:
                self.rect = self.canvas.create_rectangle(
                    self.start_x, self.start_y, event.x, event.y,
                    outline='red', width=2
                )
            
        def on_release(self, event):
            self.end_x = event.x
            self.end_y = event.y
            
            left = min(self.start_x, self.end_x)
            top = min(self.start_y, self.end_y)
            width = abs(self.end_x - self.start_x)
            height = abs(self.end_y - self.start_y)
            
            if width > 10 and height > 10:
                self.selected_region = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height
                }
                if self.root:
                    self.root.quit()
                    self.root.destroy()
            else:
                self.cancel_selection()
                
        def cancel_selection(self):
            self.selected_region = None
            if self.root:
                self.root.quit()
                self.root.destroy()

def get_region_from_user():
    print("Select region:")
    print("1. Full screen")
    print("2. Custom region")
    region_choice = input("Choice (1-2): ")
    
    if region_choice == "1":
        return {"top": 0, "left": 0, "width": 2560, "height": 1440}
    elif region_choice == "2":
        print("Drag to select your capture region...")
        print("Press ESC to cancel selection")
        
        try:
            from OpenAI_Gym_Experiments import RegionSelector
            selector = RegionSelector()
        except ImportError:
            selector = FallbackRegionSelector()
        region = selector.select_region()
        
        if region:
            print(f"Selected region: {region}")
            return region
        else:
            print("Selection cancelled, using full screen")
            return {"top": 0, "left": 0, "width": 2560, "height": 1440}
    else:
        print("Invalid choice, using full screen")
        return {"top": 0, "left": 0, "width": 2560, "height": 1440}

class MultiInstanceLauncher:
    def __init__(self):
        self.instances = []
        self.base_data_dir = "training_data"
        
    def create_instance_config(self, instance_id):
        instance_dir = f"{self.base_data_dir}_instance_{instance_id}"
        os.makedirs(instance_dir, exist_ok=True)
        
        config = {
            'instance_id': instance_id,
            'data_dir': instance_dir,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        return config
    
    def launch_data_collection(self, instance_id, region=None):
        config = self.create_instance_config(instance_id)
        
        env = os.environ.copy()
        env['INSTANCE_ID'] = str(instance_id)
        env['DATA_DIR'] = config['data_dir']
        if region:
            env['CAPTURE_REGION'] = f"{region['left']},{region['top']},{region['width']},{region['height']}"
        
        cmd = ["python", "OpenAI Gym Experiments.py", "--instance-mode", "--data-collection"]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.instances.append({
            'id': instance_id,
            'type': 'data_collection',
            'process': process,
            'config': config
        })
        
        print(f"Started data collection instance {instance_id} (PID: {process.pid})")
        return process
    
    def launch_training(self, instance_id, region=None):
        config = self.create_instance_config(instance_id)
        
        env = os.environ.copy()
        env['INSTANCE_ID'] = str(instance_id)
        env['DATA_DIR'] = config['data_dir']
        if region:
            env['CAPTURE_REGION'] = f"{region['left']},{region['top']},{region['width']},{region['height']}"
        
        cmd = ["python", "OpenAI Gym Experiments.py", "--instance-mode", "--training"]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.instances.append({
            'id': instance_id,
            'type': 'training',
            'process': process,
            'config': config
        })
        
        print(f"Started training instance {instance_id} (PID: {process.pid})")
        return process
    
    def launch_offline_training(self, instance_id, data_source_dir=None):
        config = self.create_instance_config(instance_id)
        
        env = os.environ.copy()
        env['INSTANCE_ID'] = str(instance_id)
        env['SOURCE_DATA_DIR'] = data_source_dir or self.base_data_dir
        env['OUTPUT_MODEL'] = f"assaultcube_trained_instance_{instance_id}.pth"
        
        cmd = ["python", "offline_trainer.py"]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.instances.append({
            'id': instance_id,
            'type': 'offline_training',
            'process': process,
            'config': config
        })
        
        print(f"Started offline training instance {instance_id} (PID: {process.pid})")
        return process
    
    def launch_test_model(self, instance_id, model_path, region=None):
        config = self.create_instance_config(instance_id)
        
        env = os.environ.copy()
        env['INSTANCE_ID'] = str(instance_id)
        env['MODEL_PATH'] = model_path
        if region:
            env['CAPTURE_REGION'] = f"{region['left']},{region['top']},{region['width']},{region['height']}"
        
        cmd = ["python", "test_trained_model.py"]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.instances.append({
            'id': instance_id,
            'type': 'model_test',
            'process': process,
            'config': config
        })
        
        print(f"Started model test instance {instance_id} (PID: {process.pid})")
        return process
    
    def get_status(self):
        active_instances = []
        completed_instances = []
        
        for instance in self.instances:
            if instance['process'].poll() is None:
                active_instances.append(instance)
            else:
                completed_instances.append(instance)
        
        return active_instances, completed_instances
    
    def stop_instance(self, instance_id):
        for instance in self.instances:
            if instance['id'] == instance_id and instance['process'].poll() is None:
                instance['process'].terminate()
                print(f"Stopped instance {instance_id}")
                return True
        print(f"Instance {instance_id} not found or already stopped")
        return False
    
    def stop_all(self):
        for instance in self.instances:
            if instance['process'].poll() is None:
                instance['process'].terminate()
        print("Stopped all instances")
    
    def monitor_instances(self):
        while True:
            active, completed = self.get_status()
            if active:
                print(f"\nActive instances: {len(active)}")
                for instance in active:
                    print(f"  Instance {instance['id']} ({instance['type']}) - PID: {instance['process'].pid}")
            if completed:
                print(f"Completed instances: {len(completed)}")
                for instance in completed:
                    print(f"  Instance {instance['id']} ({instance['type']}) - Exit code: {instance['process'].returncode}")
            time.sleep(10)

def main():
    launcher = MultiInstanceLauncher()
    
    print("Multi-Instance Launcher")
    print("1. Launch Data Collection Instance")
    print("2. Launch Training Instance") 
    print("3. Launch Offline Training Instance")
    print("4. Launch Model Test Instance")
    print("5. View Instance Status")
    print("6. Stop Instance")
    print("7. Stop All Instances")
    print("8. Monitor Instances")
    print("9. Exit")
    
    while True:
        choice = input("\nEnter choice (1-9): ")
        
        if choice == "1":
            instance_id = input("Enter instance ID: ")
            region = get_region_from_user()
            launcher.launch_data_collection(instance_id, region)
            
        elif choice == "2":
            instance_id = input("Enter instance ID: ")
            region = get_region_from_user()
            launcher.launch_training(instance_id, region)
            
        elif choice == "3":
            instance_id = input("Enter instance ID: ")
            data_dir = input("Data source directory (or press Enter for default): ").strip()
            if not data_dir:
                data_dir = None
            launcher.launch_offline_training(instance_id, data_dir)
            
        elif choice == "4":
            instance_id = input("Enter instance ID: ")
            model_path = input("Model path: ")
            launcher.launch_test_model(instance_id, model_path)
            
        elif choice == "5":
            active, completed = launcher.get_status()
            print(f"\nActive: {len(active)}, Completed: {len(completed)}")
            for instance in active:
                print(f"  Active - Instance {instance['id']} ({instance['type']})")
            for instance in completed:
                print(f"  Done - Instance {instance['id']} ({instance['type']}) - Exit: {instance['process'].returncode}")
                
        elif choice == "6":
            instance_id = input("Enter instance ID to stop: ")
            launcher.stop_instance(instance_id)
            
        elif choice == "7":
            launcher.stop_all()
            
        elif choice == "8":
            print("Starting monitor (Ctrl+C to stop)...")
            try:
                launcher.monitor_instances()
            except KeyboardInterrupt:
                print("\nMonitor stopped")
                
        elif choice == "9":
            launcher.stop_all()
            break
            
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
