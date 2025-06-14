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
                addr = module.addr.split('-')[0]
                print(f"  {i+1:2d}. {addr} - {module.path}")
                
                if 'hl.exe' in module.path.lower() or 'client.dll' in module.path.lower():
                    self.base_address = int(addr, 16)
                    print(f"Using base address: 0x{self.base_address:08X}")
                    return
            
            if modules:
                addr = modules[0].addr.split('-')[0]
                self.base_address = int(addr, 16)
                print(f"Using first module as base: 0x{self.base_address:08X}")
                
        except Exception as e:
            print(f"Error finding base address: {e}")
            self.base_address = 0x400000
    
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
    
    def scan_for_player_data(self):
        """Scan memory for player-like data"""
        if not self.base_address:
            print("No base address!")
            return
        
        print(f"Scanning memory around base address 0x{self.base_address:08X}")
        
        offset_patterns = [
            0x00A3ACD4,
            0x00A73FA0,
            0x009E35D4,
            0x00B3B2A4,
        ]
        
        print("Testing known offset patterns:")
        for i, offset in enumerate(offset_patterns):
            test_addr = self.base_address + offset
            value = self.read_memory(test_addr)
            
            if value is not None and isinstance(value, (int, float)):
                print(f"  Pattern {i+1}: 0x{offset:08X} -> {value}")
                
                if isinstance(value, float) and 0x400000 < value < 0x7FFFFFFF:
                    ptr_value = self.read_memory(int(value))
                    if ptr_value is not None:
                        print(f"    -> Pointer to: {ptr_value}")
            else:
                print(f"  Pattern {i+1}: 0x{offset:08X} -> (failed)")
        
        print("Brute force scanning...")
        interesting_values = []
        
        for offset in range(0, 0x100000, 4):
            addr = self.base_address + offset
            value = self.read_memory(addr)
            
            if value is not None and isinstance(value, (int, float)):
                if -10000 < value < 10000 and value != 0:
                    interesting_values.append((offset, value))
                    
                    if len(interesting_values) % 100 == 0:
                        print(f"  Scanned {offset:08X}, found {len(interesting_values)} candidates...")
        
        print(f"Found {len(interesting_values)} potentially interesting values:")
        for i, (offset, value) in enumerate(interesting_values[:20]):
            print(f"  0x{offset:08X}: {value}")
        
        return interesting_values
    
    def interactive_memory_explorer(self):
        """Let user explore memory interactively"""
        print("Interactive Memory Explorer")
        print("Enter memory addresses to read (hex format like 0x400000)")
        print("Or offsets from base address (like +0x1000)")
        print("Type 'quit' to exit")
        
        if not self.base_address:
            print("No base address available!")
            return
        
        while True:
            try:
                user_input = input("\nAddress/Offset: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.startswith('+'):
                    offset = int(user_input[1:], 16)
                    addr = self.base_address + offset
                elif user_input.startswith('0x'):
                    addr = int(user_input, 16)
                else:
                    addr = int(user_input, 16)
                
                float_val = self.read_memory(addr)
                print(f"  0x{addr:08X} (float): {float_val}")
                
                raw_bytes = self.read_memory(addr, 4)
                if raw_bytes and isinstance(raw_bytes, bytes):
                    int_val = ctypes.c_int.from_buffer(ctypes.create_string_buffer(raw_bytes)).value
                    print(f"  0x{addr:08X} (int):   {int_val}")
                
                print("  Nearby values:")
                for i in range(-2, 3):
                    nearby_addr = addr + (i * 4)
                    nearby_val = self.read_memory(nearby_addr)
                    print(f"    +{i*4:2d}: 0x{nearby_addr:08X} = {nearby_val}")
                
            except ValueError:
                print("Invalid address format. Use hex like 0x400000 or +0x1000")
            except Exception as e:
                print(f"Error: {e}")

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
    print("1. Scan for player data automatically")
    print("2. Explore memory interactively")
    print("3. Test specific offsets")
    print("4. Exit")
    
    choice = input("\nChoose (1-4): ").strip()
    
    if choice == "1":
        hunter.scan_for_player_data()
    elif choice == "2":
        hunter.interactive_memory_explorer()
    elif choice == "3":
        print("Feature coming soon!")
    
    print("Memory hunting complete!")
    print("Use the found addresses to update cs16_offsets.py")

if __name__ == "__main__":
    main()
