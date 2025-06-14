#!/usr/bin/env python3
"""
CS 1.6 Memory Offset Hunter
Find the REAL memory addresses for your CS version
"""

import psutil
import ctypes
from ctypes import wintypes
import time
import win32gui
import win32process

class CS16OffsetHunter:
    """Hunt for the real memory offsets"""
    
    def __init__(self):
        self.process = None
        self.process_handle = None
        self.base_address = None
        
    def find_cs16_process(self):
        """Find CS 1.6 process"""
        process_names = ["hl.exe", "cstrike.exe", "counter-strike.exe"]
        
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                if proc.info['name'].lower() in [name.lower() for name in process_names]:
                    print(f"Found CS 1.6 process: {proc.info['name']} (PID: {proc.info['pid']})")
                    
                    self.process = proc
                    self.process_handle = ctypes.windll.kernel32.OpenProcess(
                        0x1F0FFF,
                        False,
                        proc.info['pid']
                    )
                    
                    if self.process_handle:
                        print(f"Got process handle: {self.process_handle}")
                        self.find_base_address()
                        return True
                    else:
                        print("Failed to get process handle")
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        print("CS 1.6 process not found!")
        return False
    
    def find_base_address(self):
        """Find the base address of client.dll or hl.exe"""
        try:
            if not self.process:
                return
            modules = list(self.process.memory_maps())
            
            print(f"Found {len(modules)} memory regions:")
            
            for i, module in enumerate(modules[:10]):
                try:
                    if hasattr(module, 'addr'):
                        addr = module.addr.split('-')[0]
                    elif hasattr(module, 'start'):
                        addr = hex(module.start)[2:]
                    else:
                        addr = str(module).split()[0]
                    
                    path = getattr(module, 'path', str(module))
                    print(f"  {i+1:2d}. {addr} - {path}")
                    
                    if 'hl.exe' in path.lower() or 'client.dll' in path.lower():
                        self.base_address = int(addr, 16)
                        print(f"Using base address: 0x{self.base_address:08X}")
                        return
                except Exception as mod_err:
                    print(f"  {i+1:2d}. Error reading module: {mod_err}")
                    continue
            
            if modules:
                try:
                    first_module = modules[0]
                    if hasattr(first_module, 'addr'):
                        addr = first_module.addr.split('-')[0]
                    elif hasattr(first_module, 'start'):
                        addr = hex(first_module.start)[2:]
                    else:
                        addr = "400000"
                    self.base_address = int(addr, 16)
                    print(f"Using first module as base: 0x{self.base_address:08X}")
                except Exception:
                    self.base_address = 0x400000
                    print("Using default base address: 0x00400000")
                
        except Exception as e:
            print(f"Error finding base address: {e}")
            self.base_address = 0x400000
            print("Using default base address: 0x00400000")
    
    def read_memory(self, address, size=4):
        """Read memory from process"""
        try:
            buffer = ctypes.create_string_buffer(size)
            bytes_read = ctypes.c_size_t()
            
            success = ctypes.windll.kernel32.ReadProcessMemory(
                self.process_handle,
                ctypes.c_void_p(address),
                buffer,
                size,
                ctypes.byref(bytes_read)
            )
            
            if success and bytes_read.value == size:
                if size == 4:
                    return ctypes.c_float.from_buffer(buffer).value
                else:
                    return buffer.raw
            return None
            
        except Exception:
            return None
    
    def find_player_coordinates_practical(self):
        """Find player coordinates by watching for changing float values"""
        if not self.process_handle:
            print("No process handle!")
            return
        
        print("PRACTICAL COORDINATE FINDER")
        print("This will scan for float values that change when you move")
        print("1. Stand still in CS 1.6")
        print("2. Press Enter to take first snapshot")
        input("Press Enter when ready...")
        
        snapshot1 = self.scan_float_values()
        if not snapshot1:
            print("Failed to scan memory")
            return
        
        print(f"Found {len(snapshot1)} float values")
        print("Now MOVE in CS 1.6 (walk forward/backward)")
        input("Press Enter after moving...")
        snapshot2 = self.scan_float_values()
        if not snapshot2:
            print("Failed to scan memory")
            return
        
        changed_values = []
        for addr, val1 in snapshot1.items():
            if addr in snapshot2:
                val2 = snapshot2[addr]
                if abs(val1 - val2) > 0.01:
                    changed_values.append((addr, val1, val2, abs(val1 - val2)))
        
        if changed_values:
            print(f"\nFound {len(changed_values)} values that changed:")
            changed_values.sort(key=lambda x: x[3], reverse=True)
            
            for i, (addr, val1, val2, diff) in enumerate(changed_values[:10]):
                print(f"  {i+1:2d}. 0x{addr:08X}: {val1:8.3f} -> {val2:8.3f} (diff: {diff:6.3f})")
            
            print("\nThese addresses might contain player coordinates!")
            print("Test them by reading these addresses while moving around")
        else:
            print("No significantly changed values found")
            print("Try moving more or ensure CS 1.6 window is active")
    
    def scan_float_values(self):
        """Scan memory for float values in reasonable ranges"""
        values = {}
        
        try:
            for addr in range(0x400000, 0x800000, 4):
                value = self.read_memory(addr, 4)
                if value is not None and isinstance(value, (int, float)):
                    if -10000 < value < 10000 and abs(value) > 0.001:
                        values[addr] = value
                        
                if len(values) > 50000:
                    break
                    
        except Exception as e:
            print(f"Scan error: {e}")
        
        return values

def main():
    print("CS 1.6 Memory Offset Hunter")
    print("=" * 50)
    print("This will help find the REAL memory addresses!")
    print()
    print("REQUIREMENTS:")
    print("- CS 1.6 must be running")
    print("- You should be in-game (not menu)")
    print("- Run this as administrator for best results")
    print()
    
    input("Press Enter when CS 1.6 is running...")
    
    hunter = CS16OffsetHunter()
    
    if not hunter.find_cs16_process():
        print("Can't find CS 1.6. Make sure it's running!")
        return
    
    print("WHAT DO YOU WANT TO DO:")
    print("1. Find player coordinates (PRACTICAL - BEST OPTION)")
    print("2. Exit")
    
    choice = input("\nChoose (1-2): ").strip()
    
    if choice == "1":
        hunter.find_player_coordinates_practical()
    elif choice == "2":
        return
    else:
        print("Invalid choice")
        
    print("Memory hunting complete!")
    print("Use the found addresses to update cs16_offsets.py")

if __name__ == "__main__":
    main()
