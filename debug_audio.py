"""
Audio Device Debugger for WARYON
Diagnose and fix microphone detection issues
"""

import sounddevice as sd
import numpy as np

def debug_audio_devices():
    """Debug audio device detection issues"""
    print("🔍 AUDIO DEVICE DEBUGGING")
    print("=" * 40)
    
    try:
        # Get all devices
        devices = sd.query_devices()
        print(f"📱 Total devices found: {len(devices)}")
        print()
        
        # List ALL devices with details
        input_devices = []
        output_devices = []
        
        for i, device in enumerate(devices):
            device_info = f"Device {i}: {device['name']}"
            
            # Check device type
            max_inputs = device.get('max_inputs', 0)
            max_outputs = device.get('max_outputs', 0)
            
            print(f"{device_info}")
            print(f"  Inputs: {max_inputs}, Outputs: {max_outputs}")
            print(f"  Sample Rate: {device.get('default_samplerate', 'Unknown')}")
            print(f"  Host API: {device.get('hostapi', 'Unknown')}")
            print()
            
            if max_inputs > 0:
                input_devices.append((i, device))
            if max_outputs > 0:
                output_devices.append((i, device))
        
        print(f"🎤 Input devices found: {len(input_devices)}")
        print(f"🔊 Output devices found: {len(output_devices)}")
        
        if not input_devices:
            print("❌ NO INPUT DEVICES FOUND!")
            print("\n🔧 TROUBLESHOOTING STEPS:")
            print("1. Check microphone is plugged in")
            print("2. Check Windows microphone permissions")
            print("3. Try different microphone/headset")
            print("4. Check Windows Sound settings")
            return False
        
        # Test default input device
        try:
            default_input = sd.default.device[0] if sd.default.device else None
            print(f"\n🎯 Default input device: {default_input}")
            
            # Try to record from default device
            print("🧪 Testing 2-second recording from default device...")
            
            test_recording = sd.rec(
                int(2 * 44100),  # 2 seconds at 44100 Hz
                samplerate=44100,
                channels=1,
                dtype=np.float32
            )
            sd.wait()
            
            # Analyze recording
            max_volume = np.max(np.abs(test_recording))
            avg_volume = np.mean(np.abs(test_recording))
            
            print(f"✅ Recording successful!")
            print(f"   Max volume: {max_volume:.6f}")
            print(f"   Avg volume: {avg_volume:.6f}")
            
            if max_volume > 0.0001:
                print("✅ Microphone input detected - READY FOR WARYON!")
                return True
            else:
                print("⚠️ Very quiet input - check microphone level")
                print("   Try speaking louder or adjusting mic volume")
                return True  # Still working, just quiet
                
        except Exception as record_error:
            print(f"❌ Recording test failed: {record_error}")
            
            # Try manual device selection
            print("\n🔧 Trying manual device selection...")
            for device_id, device in input_devices[:3]:  # Try first 3
                try:
                    print(f"Testing device {device_id}: {device['name']}")
                    
                    test_rec = sd.rec(
                        int(1 * 44100),  # 1 second test
                        samplerate=44100,
                        channels=1,
                        device=device_id,
                        dtype=np.float32
                    )
                    sd.wait()
                    
                    vol = np.max(np.abs(test_rec))
                    print(f"   Volume: {vol:.6f}")
                    
                    if vol > 0.0001:
                        print(f"✅ Working device found: {device_id}")
                        return device_id
                        
                except Exception as dev_error:
                    print(f"   Failed: {dev_error}")
                    continue
        
        return False
        
    except Exception as e:
        print(f"❌ Debug error: {e}")
        return False

def fix_audio_processor():
    """Generate fixed audio processor code"""
    working_device = debug_audio_devices()
    
    if working_device is False:
        print("\n❌ No working audio devices found")
        print("🔧 WARYON will run with mock audio for competition demo")
        return
    
    # Create fixed audio processor
    print(f"\n🔧 Creating fixed audio processor...")
    
    if isinstance(working_device, int):
        device_override = f"self.device_id = {working_device}  # Working device found"
        device_param = f"device={working_device},"
    else:
        device_override = "# Using default device"
        device_param = ""
    
    fixed_code = f'''
# FIXED AUDIO PROCESSOR FOR WARYON
# Add this to your audio_processor.py __init__ method:

def __init__(self, config_manager, threat_callback: Callable = None):
    # ... existing init code ...
    
    # DEVICE FIX
    {device_override}
    
    print("🎤 Real Audio Processor initialized with device fix")

# FIXED test_audio method:
def test_audio(self) -> bool:
    """Test real microphone access with device fix"""
    try:
        print("🧪 Testing microphone with device fix...")
        
        # Test recording with device override
        test_recording = sd.rec(
            int(2 * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            {device_param}
            dtype=np.float32
        )
        sd.wait()
        
        max_volume = np.max(np.abs(test_recording))
        print(f"✅ Fixed microphone test - Volume: {{max_volume:.6f}}")
        
        return max_volume > 0.0001
        
    except Exception as e:
        print(f"❌ Fixed test failed: {{e}}")
        return False
'''
    
    print("📝 Audio fix code generated above!")
    print("✅ Copy this fix into your audio_processor.py")

if __name__ == "__main__":
    fix_audio_processor()