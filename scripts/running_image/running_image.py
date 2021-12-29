#!/usr/bin/python3
import numpy as np
import argparse
import csv
import cv2


# Define params (this time without argparse)
FRAMES_PER_PIXEL = 2


# Create argument parser
parser = argparse.ArgumentParser(description='Generate pulsing heart effect.')
parser.add_argument('coordinates_csv_path', type=str,
                    help='Path to LED coordinates csv')
parser.add_argument('image_path', type=str,
                    help='Path to image')
parser.add_argument('--frames_per_pixel', type=int, nargs='?', default=2,
                    help='Number of frames per pixel')
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

# Find tree center
tree_height = np.max(coords, axis=0)[2]
tree_center = np.array([0.0, 0.0, tree_height/2.0])

# Read in image
img_raw = cv2.imread(args.image_path)
img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)   # BGR -> RGB
gift_to_px = img.shape[0]/tree_height
window_radius_px = int(round(gift_to_px))
window_center_start = window_radius_px
window_center_end = img.shape[1]-window_radius_px

frame_id = 0
for center in range(window_center_start, window_center_end):
    subimage = img[:, center-window_radius_px:center+window_radius_px]
    for frame in range(args.frames_per_pixel):
        led_matrix = np.zeros((coords.shape[0], 3))
        for led_i in range(led_matrix.shape[0]):
            led_y, led_z = coords[led_i, 1:]
            px_x = max(0, min(subimage.shape[1]-1, int(round(subimage.shape[1]/2.0+led_y*gift_to_px))))
            px_y = subimage.shape[0]-1-max(0, min(subimage.shape[0]-1, int(round(led_z*gift_to_px))))
            led_matrix[led_i, :] = subimage[px_y, px_x]
        led_vector = led_matrix.flatten()
        led_row = [frame_id]
        led_row.extend(led_vector.tolist())
        output_data.append(led_row)
        frame_id += 1

# Write CSV
with open('running_image.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(output_data)
