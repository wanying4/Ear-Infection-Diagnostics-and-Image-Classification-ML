#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Wanying Li
# Created Date: May 15, 2017
# ---------------------------------------------------------------------------
"""
This is the Python code running on the Raspberry Pi that captures & sends the image, then receives & outputs the diagnosis results.
The Raspberry Pi is setup with a Spy Picamera, LEDs and screen monitor.
""" 
# ---------------------------------------------------------------------------


import time
from time import sleep
from picamera import PiCamera
from gpiozero import Button
import RPi.GPIO as GPIO
import requests

button = Button(18) # pin 18 for the reset button
camera = PiCamera()

# Define the pins used
PIN_img_capture = 17 # pin 17 for the capture image button
PIN_green_LED = 5 # pin 5 for green LED (ie. normal eardrum)
PIN_red_LED = 6 # pin 6 for red LED (ie. infected eardrum)

# Initiatize the GPIO pins
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_img_capture, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_green_LED,GPIO.OUT)
GPIO.setup(PIN_red_LED,GPIO.OUT)

# Camera specs
camera.resolution = (800,600)
camera.brightness = 70
camera.contrast = 70
print('Camera initialization is done.')

def main():
    # Trigger camera livestream
    GPIO.wait_for_edge(PIN_img_capture, GPIO.FALLING)
    print("Button pressed for liveview")
    time.sleep(0.2)
    camera.start_preview() # livestream pops up after .2 seconds
    
    while GPIO.input(PIN_img_capture) == GPIO.LOW:
        time.sleep(0.01)
    
    # Take the image when the capture button is pressed the 2nd time and turn off the livestream
    if GPIO.input(PIN_img_capture) == GPIO.HIGH:
        print("Button pressed to capture image")
        # Take pic
        camera.capture('/home/pi/Desktop/PiCam_image.jpg') # capture and save image
        camera.stop_preview()
    
    # Upload the picture taken from the camera to the local server and get the result
    url = 'http://c0a67e54.ngrok.io/classification' #change ngorok accordingly
    filepath = '/home/pi/Desktop/PiCam_image.jpg'
    files = {'file': open(filepath, 'rb')}
    print('Waiting for results...')
    start = time.time()
    r = requests.post(url, files=files)
    results = r.text
    length = time.time() - start
    
    # Display the result and the time the algorithm had taken in the console
    print('The eardrum is {}. The algorithm used {:.1f} seconds.'.format(results, length))
    
    if 'normal' in results:
        GPIO.output(PIN_green_LED,GPIO.HIGH) # turn GPIO pin on
        time.sleep(3)
        GPIO.output(PIN_green_LED,GPIO.LOW) # turn GPIO pin off
    else:
        GPIO.output(PIN_red_LED,GPIO.HIGH) # turn GPIO pin on
        time.sleep(3)
        GPIO.output(PIN_red_LED,GPIO.LOW) # turn GPIO pin off

main()

# Push the rest button to run the algorithm again
while 0==0:
    button.wait_for_press()
    print('Reset')
    time.sleep(0.5)
    main()