import RPi.GPIO as GPIO
import time

# Set GPIO mode to BCM
GPIO.setmode(GPIO.BCM)

# Define GPIO pins for Trigger and Echo
GPIO_TRIGGER = 23
GPIO_ECHO = 24

# Set up the direction of the pins (In/Out)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def get_distance():
    # 1. Send a 10-microsecond pulse to trigger the sensor
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    StartTime = time.time()
    StopTime = time.time()

    # 2. Record the time when the ECHO pin goes HIGH (pulse sent)
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

    # 3. Record the time when the ECHO pin goes LOW (pulse returned)
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()

    # 4. Calculate the time difference
    TimeElapsed = StopTime - StartTime

    # 5. Calculate distance
    # Speed of sound is roughly 34,300 cm/s.
    # We divide by 2 because the sound wave travels to the object AND back.
    distance = (TimeElapsed * 34300) / 2

    return distance

if __name__ == '__main__':
    try:
        print("Ultrasonic Measurement - Press CTRL+C to stop")
        # Give the sensor a moment to settle
        time.sleep(2) 
        
        while True:
            dist = get_distance()
            print(f"Measured Distance = {dist:.1f} cm")
            time.sleep(1) # Wait 1 second before the next measurement

    # Handle the user pressing CTRL+C to stop the script gracefully
    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
        GPIO.cleanup()