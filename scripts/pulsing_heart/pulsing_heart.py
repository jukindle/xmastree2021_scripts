#!/usr/bin/python3
import numpy as np
import argparse
import csv


# Create argument parser
parser = argparse.ArgumentParser(description='Generate pulsing heart effect.')
parser.add_argument('coordinates_csv_path', type=str,
                    help='Path to LED coordinates csv')
parser.add_argument('--beat_frame_rate', type=int, nargs='?', default=70,
                    help='Rate at which heart beats (frames per full beat cycle)')
parser.add_argument('--beat_to_silence_ratio', type=float, nargs='?', default=0.7,
                    help='Ratio between beat and silence')
parser.add_argument('--heart_radius', type=float, nargs='?', default=1.4,
                    help='Radius of the heart')
parser.add_argument('--led_radius', type=float, nargs='?', default=0.02,
                    help='Radius of each LED in GIFT space')
parser.add_argument('--amplitude_small', type=float, nargs='?', default=0.6,
                    help='Amplitude of prior and posterior beat')
parser.add_argument('--beat_overlap', type=float, nargs='?', default=0.3,
                    help='Overlap of prior&posterior to main beat')
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

# Find heart center
heart_center = np.array([0.0, 0.0,  np.max(coords, axis=0)[2]/2.0])

# Define beat formula
a_small = args.amplitude_small
beat_overlap = args.beat_overlap
fh_raw = lambda x: np.cos((x-0.5)*2*np.pi)*0.5+0.5 if (x > 0 and x < 1) else 0.0
fh_sup = lambda x: a_small*fh_raw(x) + fh_raw(x-(1.0-beat_overlap)) + a_small*fh_raw(x-2.0*(1.0-beat_overlap))
fh = lambda x: fh_sup(x*(2.0*(1.0-beat_overlap)+1))

# Debug
# xx = np.linspace(0, 6, 1000)
# y = [fh(x) for x in xx]
# import matplotlib.pyplot as plt
# plt.plot(xx, y)
# plt.show()

# Create frames for a single beat
rgb_scaling = np.array([[1], [0], [0]]).T
distances = np.linalg.norm(coords - heart_center, axis=1, keepdims=True)
for frame in range(args.beat_frame_rate):
    thresh_value = args.heart_radius*fh(frame/args.beat_frame_rate/args.beat_to_silence_ratio)
    threshed_leds = distances < thresh_value + args.led_radius

    led_matrix = (threshed_leds*254).dot(rgb_scaling)
    led_vector = led_matrix.flatten()

    led_row = [frame]
    led_row.extend(led_vector.tolist())
    output_data.append(led_row)


# Write CSV
with open('pulsing_heart.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(output_data)
