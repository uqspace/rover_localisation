# AURC 2024 - Autonomous Mapping Challenge
# Localisation
Take photos of flat ChArUco grid with the camera from many orientations, distances and angles and collate in a folder named ‘calib’ in the same directory as the calibration program.

Run aruco_calibration.py. If successful camera_matrix.npy and dist_coeffs.npy files will be generated. If failed the error message will indicate which index image was too poor to process - manually remove the image and try again. Warning it is fairly sensitive. 

ArUco markers must be of ID’s 0 onwards and be orientated towards the camera from the perspective of the rover driving away. Check the marker orientation by running pose_detection2.py and finding the orientation where the angle its approximately 0 degrees when held flat.

Run pose_detection.py to begin localisation. The 0 marker will indicate (0,0) and should be immediately deployed upon starting.

The camera should be able to see one marker at all times and must see the previous marker and the new marker in the same frame briefly to be able to calculate its location based on the newly deployed marker.
Press ‘q’ to end localisation and display map

# Mapping
Look at the camera feed and add unity assets into the scene.
