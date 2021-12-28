#!/usr/bin/python3
import numpy as np
import argparse
import csv


# Define params (this time without argparse)
# Acceleration params
LED_RADIUS = 0.25
FREQ_X = 10.0
FREQ_Y = 10.0
AMP_X = 0.45
AMP_Y = 0.45
AMP_Z = 1.4
Z_DEP_SCALING = 0.5
PARTICLE_1_COLOR = np.array([[1], [0], [0]]).T
# For all colorblindies like me, red and blue instead of red and green :)
PARTICLE_2_COLOR = np.array([[0], [0], [1]]).T
FRAMES_ACCELERATION = 500
ACCELERATION_REPEAT_CYCLES = 4.5
TIME_SCALING_ACCELERATION = 0.2

# Empty frames params
EMPTY_FRAMES = 10

# Explosion params
EXPLOSION_COLOR = np.array([[1], [0], [1]]).T
EXPLOSION_SPEED = 1.0
COOLDOWN_SPEED = 0.5
FRAMES_EXPLOSION = 10
FRAMES_COOLDOWN = 360


# Create argument parser
parser = argparse.ArgumentParser(description='Generate pulsing heart effect.')
parser.add_argument('coordinates_csv_path', type=str,
                    help='Path to LED coordinates csv')
args = parser.parse_args()

# Read in calibration
coords = np.genfromtxt(args.coordinates_csv_path, delimiter=',')
with open(args.coordinates_csv_path, 'r', encoding='utf-8-sig') as f:
    coords = np.genfromtxt(f, dtype=float, delimiter=',')


# Prepare output data with header
output_data = []
header = ['FRAME_ID']
for i in range(coords.shape[0]):
    header.append("R_{}".format(i))
    header.append("G_{}".format(i))
    header.append("B_{}".format(i))
output_data.append(header)

# Find explosion center
explosion_center = np.array([0.0, 0.0,  np.max(coords, axis=0)[2]/2.0])

# Define particle acceleration formula
def f_accel(t, face=1):
    z = face*AMP_Z*np.sin(2*np.pi*t)
    y = face*(1.0-Z_DEP_SCALING*z)*AMP_Y*np.cos(FREQ_Y*2*np.pi*t)
    x = face*(1.0-Z_DEP_SCALING*z)*AMP_X*np.sin(FREQ_X*2*np.pi*t)
    return np.array([x, y, z])
f_timescaling = lambda t: t*t*TIME_SCALING_ACCELERATION

# Create frames for acceleration
frame_id = 0
for frame in range(FRAMES_ACCELERATION):
    particle_pos_1 = explosion_center + f_accel(f_timescaling(ACCELERATION_REPEAT_CYCLES*frame/FRAMES_ACCELERATION), 1)
    particle_pos_2 = explosion_center + f_accel(f_timescaling(ACCELERATION_REPEAT_CYCLES*frame/FRAMES_ACCELERATION), -1)
    distances_1 = np.linalg.norm(coords - particle_pos_1, axis=1, keepdims=True)
    distances_2 = np.linalg.norm(coords - particle_pos_2, axis=1, keepdims=True)
    threshed_leds_1 = distances_1 < LED_RADIUS
    threshed_leds_2 = distances_2 < LED_RADIUS
    led_vector_1 = (threshed_leds_1*254).dot(PARTICLE_1_COLOR).flatten()
    led_vector_2 = (threshed_leds_2*254).dot(PARTICLE_2_COLOR).flatten()
    led_vector = np.maximum(led_vector_1, led_vector_2)
    # led_vector[led_vector == 0] = 20

    led_row = [frame_id]
    led_row.extend(led_vector.tolist())
    output_data.append(led_row)
    frame_id += 1

# Create empty frames
for frame in range(EMPTY_FRAMES):
    led_row = [frame_id]
    led_row.extend(np.zeros((coords.shape[0]*3,)).tolist())
    output_data.append(led_row)
    frame_id += 1


# Define explosion and cooldown formula
f_explosion_radius = lambda t: t
f_cooldown_intensity = lambda t: 1.0 - t if t < 1.0 else 0.0

# Create explosion frames
for frame in range(FRAMES_EXPLOSION):
    explosion_radius = f_explosion_radius(frame/FRAMES_EXPLOSION*EXPLOSION_SPEED)
    distances = np.linalg.norm(coords - explosion_center, axis=1, keepdims=True)
    threshed_leds = distances < explosion_radius
    led_vector = (threshed_leds*254).dot(EXPLOSION_COLOR).flatten()
    # led_vector[led_vector == 0] = 20

    led_row = [frame_id]
    led_row.extend(led_vector.tolist())
    output_data.append(led_row)
    frame_id += 1

for frame in range(FRAMES_COOLDOWN):
    cooldown_intensity = f_cooldown_intensity(frame/FRAMES_COOLDOWN)
    threshed_leds = np.ones((coords.shape[0], 1))
    led_vector = cooldown_intensity*(threshed_leds*254).dot(EXPLOSION_COLOR).flatten()
    # led_vector[led_vector == 0] = 20

    led_row = [frame_id]
    led_row.extend(led_vector.tolist())
    output_data.append(led_row)
    frame_id += 1




# Write CSV
with open('particle_accelerator.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(output_data)
