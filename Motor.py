from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

# --- Left Motor Setup ---
ena = PWMOutputDevice(20, frequency=1000) 
in1 = DigitalOutputDevice(26) 
in2 = DigitalOutputDevice(19) 

# --- Right Motor Setup ---
enb = PWMOutputDevice(5, frequency=1000)  
in3 = DigitalOutputDevice(13) 
in4 = DigitalOutputDevice(6)

# --- Control Functions ---

def forward(speed, duration):
    print(f"Moving forward at {speed*100}% speed")
    # Set Directions: Left Forward, Right Forward
    in1.off()
    in2.on()
    in3.off()
    in4.on()
    
    # Apply Throttle
    ena.value = speed
    enb.value = speed
    
    sleep(duration)
    stop()

def backward(speed, duration):
    print("Moving backward")
    # Set Directions: Left Backward, Right Backward
    in1.on()
    in2.off()
    in3.on()
    in4.off()
    
    # Apply Throttle
    ena.value = speed
    enb.value = speed
    
    sleep(duration)
    stop()

def turn_left(speed, duration):
    print("Turning left")
    # Set Directions: Left Backward, Right Forward
    in1.on()
    in2.off()
    in3.off()
    in4.on()
    
    # Apply Throttle
    ena.value = speed
    enb.value = speed
    
    sleep(duration)
    stop()

def turn_right(speed, duration):
    print("Turning right")
    # Set Directions: Left Forward, Right Backward
    in1.off()
    in2.on()
    in3.on()
    in4.off()
    
    # Apply Throttle
    ena.value = speed
    enb.value = speed
    
    sleep(duration)
    stop()

def stop():
    # Cut throttle first
    ena.value = 0
    enb.value = 0
    # Reset all directional logic to off
    in1.off()
    in2.off()
    in3.off()
    in4.off()

# --- Main Execution ---

if __name__ == "__main__":
    try:
        print("Starting Robot...")
        
        # Execute your specific command
        turn_left(0.9, 1.2)
        
    except KeyboardInterrupt:
        print("\nEmergency Stop triggered by user")
    finally:
        stop()
        print("Motors stopped safely. GPIO cleaned up.")