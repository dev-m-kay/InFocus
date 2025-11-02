import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Speeds up OpenCV2 attaching to the a webcamera.
import mediapipe as mp
import numpy as np
import cv2

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def gaze(frame, points):

	image_points = np.array([
        relative(points.landmark[4], frame.shape),    # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),   # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)    # Right mouth corner
    ], dtype="double")

	model_points = np.array([
        (0.0, 0.0, 0.0),       # Nose tip
        (0, -63.6, -12.5),     # Chin
        (-43.3, 32.7, -26),    # Left eye left corner
        (43.3, 32.7, -26),     # Right eye right corner
        (-28.9, -28.9, -24.1), # Left Mouth corner
        (28.9, -28.9, -24.1)   # Right mouth corner
    ])

	image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")


	Eye_ball_center_right = np.array([[-29.05],[32.7],[-39.5]])
	Eye_ball_center_left = np.array([[29.05],[32.7],[-39.5]])

	focal_length = frame.shape[1]
	center = (frame.shape[1] / 2, frame.shape[0] / 2)
	camera_matrix = np.array(
		[[focal_length, 0, center[0]],
			[0, focal_length, center[1]],
			[0, 0, 1]], dtype="double"
	)

	dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
																	dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

	left_pupil = relative(points.landmark[468], frame.shape)
	right_pupil = relative(points.landmark[473], frame.shape)

	_ ,transformation, _ = cv2.estimateAffine3D(image_points1, model_points) # image cord to world cord tramsformation
	
	if transformation is not None:  
    # project pupil image point into 3d world point 

		# Left Eye

		pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

		# 3D gaze point (10 is arbitrary value denoting gaze distance)
		S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10

		# Project a 3D gaze direction onto the image plane.
		(eye_pupil2D, _) = cv2.projectPoints((int(S[0].item()), int(S[1].item()), int(S[2].item())), rotation_vector,
												translation_vector, camera_matrix, dist_coeffs)
		# Project 3D head pose into the image plane
		(head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0].item()), int(pupil_world_cord[1].item()), int(40)),
											rotation_vector,
											translation_vector, camera_matrix, dist_coeffs)
		
		# Account for Head Rotation
		gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)

		# Drawing Gaze Line
		p1 = (int(left_pupil[0]), int(left_pupil[1]))
		p2 = (int(gaze[0]), int(gaze[1]))
		cv2.line(frame, p1, p2, (0, 0, 255), 2)
	

		# Right Eye

		# project pupil image point into 3d world point 
		pupil_world_cord = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

		# 3D gaze point (10 is arbitrary value denoting gaze distance)
		S = Eye_ball_center_right + (pupil_world_cord - Eye_ball_center_right) * 10

		# Project a 3D gaze direction onto the image plane.
		(eye_pupil2D, _) = cv2.projectPoints((int(S[0].item()), int(S[1].item()), int(S[2].item())), rotation_vector,
												translation_vector, camera_matrix, dist_coeffs)
		
		# Project 3D head pose into the image plane
		(right_head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0].item()), int(pupil_world_cord[1].item()), int(40)),
											rotation_vector,
											translation_vector, camera_matrix, dist_coeffs)
		
		# Account for Head Rotation
		right_gaze = right_pupil + (eye_pupil2D[0][0] - right_pupil) - (right_head_pose[0][0] - right_pupil)

		# Drawing Gaze Line
		p1 = (int(right_pupil[0]), int(right_pupil[1]))
		p2 = (int(right_gaze[0]), int(right_gaze[1]))
		cv2.line(frame, p1, p2, (0, 0, 255), 2)


		mean_x = (right_pupil[0] + left_pupil[0])/2
		mean_y = (right_pupil[1] + left_pupil[1])/2
		gaze_mean_x = (right_gaze[0] + gaze[0])/2
		gaze_mean_y = (right_gaze[1] + gaze[1])/2

		p1 = (int(mean_x), int(mean_y))
		p2 = (int(gaze_mean_x), int(gaze_mean_y))
		cv2.line(frame, p1, p2, (0, 255, 255), 5)


mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


with mp_face_mesh.FaceMesh(
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5) as face_mesh:

	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignore Empty Camera")
			continue

		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.flip(image, 1)
		results = face_mesh.process(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		if results.multi_face_landmarks:
			gaze(image, results.multi_face_landmarks[0])  # gaze estimation

		cv2.imshow('output window', image)
		if cv2.waitKey(2) & 0xFF == 27:
			break

cap.release()

