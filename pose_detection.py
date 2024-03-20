import cv2 as cv
import numpy as np
import math
from polar_point_map import generate_map
import subprocess
from time import sleep
import threading
import matplotlib.pyplot as plt


# Load calibration data (camera matrix and distortion coefficients)
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Initialize ArUco dictionary and parameters
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters =  cv.aruco.DetectorParameters()
camera_elevation = 0.35

rover_gnd_travel = 0
markerx_separation = []
markery_separation = []
markerP_seperation = []
markerO_seperation = []
highest_marker = 0
repeated = 0
round_highest_id = 0
round_highest = 0
cap = cv.VideoCapture(0)

number_colors = {
    0: (255, 0, 0),    # Blue
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Red
    3: (255, 255, 0),  # Yellow
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Cyan
    6: (128, 0, 0),    # Maroon
    7: (0, 128, 0),    # Green (dark)
    8: (0, 0, 128),    # Navy
    9: (128, 0, 128)   # Purple
}
points = [(0,0)]
fig = None
ax = None
current_marker_dist = 0
last_knowns = {}
last_knowns2 = {}

def polar_to_cartesian(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

# Function to convert Cartesian coordinates to polar coordinates
def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta

# Function to add two polar coordinates
def add_polar(r1, theta1, r2, theta2):
    # Convert polar coordinates to Cartesian coordinates
    x1, y1 = polar_to_cartesian(r1, theta1)
    x2, y2 = polar_to_cartesian(r2, theta2)

    # Add Cartesian coordinates
    x_sum = x1 + x2
    y_sum = y1 + y2

    # Convert sum back to polar coordinates
    r_sum, theta_sum = cartesian_to_polar(x_sum, y_sum)
    return r_sum, theta_sum

def calculate_orientation(rotation_matrix):
    # Extract angles from rotation matrix
    theta_x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    theta_y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    theta_z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angles to degrees
    theta_x_deg = math.degrees(theta_x)
    theta_y_deg = math.degrees(theta_y)
    theta_z_deg = math.degrees(theta_z)

    return theta_x_deg, theta_y_deg, theta_z_deg

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict,
                                                              parameters=parameters)
    rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05,
                                                         camera_matrix,
                                                         dist_coeffs)
    cv.putText(frame, f"{rover_gnd_travel}", (20, 400),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    #print("Rover gnd distance: ", rover_gnd_travel)
    if ids is not None and tvecs is not None:
        round_highest = 0
        round_data = []
        round_data2 = []
        for i, marker_id in enumerate(ids):
            cv.aruco.drawDetectedMarkers(frame, corners)
            distance = 1.91/0.75*np.linalg.norm(tvecs[i])
            gnd_distance = math.sqrt(abs(distance**2 - camera_elevation**2))
            rotation_matrix, _ = cv.Rodrigues(rvecs[i])
            _, _, angle_deg = calculate_orientation(rotation_matrix)
            if angle_deg > 0:
                angle_deg *= 1.15
            else:
                angle_deg *=0.85
            angle_rad = math.radians(angle_deg)
            marker_x = gnd_distance * math.sin(angle_rad)
            marker_y = gnd_distance * math.cos(angle_rad)
            cv.putText(frame, f"Marker {marker_id}: {gnd_distance:.2f}m {angle_deg:.2f} degrees", (20, 40*(i+1)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, number_colors[i], 2)
            cv.putText(frame,f"Coords: {marker_x:.2f}, {marker_y:.2f}",(20, 40*(i+1) + 20),cv.FONT_HERSHEY_SIMPLEX, 0.6, number_colors[i], 2)
            round_data.append((marker_id, (marker_x, marker_y)))
            round_data2.append((marker_id, (gnd_distance, angle_rad)))
            last_knowns[int(marker_id)] = (marker_x, marker_y)
        round_highest_id = int(max(round_data, key=lambda x: x[0])[0])
        for pair in round_data:
            if pair[0] == round_highest_id:
                round_highest = pair[1]
        if round_highest_id > highest_marker:
            repeated += 1
        past_x = 0
        past_y = 0
        if round_highest_id <= highest_marker:
            for k in range(round_highest_id):
                past_x += markerx_separation[k]
                past_y += markery_separation[k]
        rover_gnd_travel = math.sqrt((round_highest[0] + past_x)**2 + (round_highest[1] + past_y)**2)
        if round_highest_id > highest_marker and repeated > 10:
            if round_highest_id-1 in last_knowns:
                print("New marker spotted!")
                diffx = last_knowns[round_highest_id - 1][0] - round_highest[0]
                diffy = last_knowns[round_highest_id - 1][1] - round_highest[1]
                if not diffy:
                    diffy = 0.001
                markerx_separation.append(diffx)
                markery_separation.append(diffy)
                markerP_seperation.append(math.sqrt(diffx**2 + diffy**2))
                markerO_seperation.append(math.atan(diffx/diffy))
                print('saved cart. coords', markerx_separation, markery_separation)
                print('saved polar coords', markerP_seperation, markerO_seperation)
                #print(marker_separation)
                highest_marker += 1
                repeated = 0

    cv.imshow('Frame', frame)
    # Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        diffx = last_knowns[round_highest_id][0]
        diffy = last_knowns[round_highest_id][1]
        if not diffy:
            diffy = 0.001
        markerx_separation.append(diffx)
        markery_separation.append(diffy)
        markerP_seperation.append(math.sqrt(diffx ** 2 + diffy ** 2))
        markerO_seperation.append(math.atan(diffx / diffy))

        with open("points.txt", "w") as file:
            file.write("0 0\n")
            for i in range(len(markerx_separation)):
                x_sum = sum(markerx_separation[:i + 1])
                y_sum = sum(markery_separation[:i + 1])
                file.write(f"{round(x_sum, 2)} {round(y_sum, 2)}\n")
        with open("points2.txt", "w") as file:
            file.write("0 0\n")
            total_skew = 0
            for i in range(len(markerP_seperation)):
                total_skew += markerO_seperation[i]
                file.write(f"{round(markerP_seperation[i], 2)} {round(total_skew, 2)}\n")
        break

cap.release()
cv.destroyAllWindows()
generate_map()