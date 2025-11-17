import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Speeds up OpenCV2 attaching to the a webcamera.
import mediapipe as mp
import numpy as np
from sklearn.linear_model import Ridge
import cv2

class GazeDisplayTracker:

	

	def __init__(self):
		self.relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
		self.relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)
		self.model = Ridge()
		self.RELATIVE_SAMPLE_POINTS = [(0.25, 0.20),(0.50, 0.20),(0.75, 0.20),(0.25, 0.50),(0.50, 0.50),(0.75, 0.50),(0.25, 0.80),(0.50, 0.80),(0.75, 0.80)]

		self.last_sample = []
		self.data_points = []
		self.target_points = []

		self.features = []

		self.hasCalibrated = False

	def generateSamplePoints(self, x_max, y_max):
		self.x_max = x_max
		self.y_max = y_max
		self.sample_points = []
		self.data_points = []
		self.target_points = []

		for rsp in self.RELATIVE_SAMPLE_POINTS:
			self.sample_points.append((int(rsp[0]*x_max), int(rsp[1]*y_max)))


		return self.sample_points

	def get_sample(self, point_index):
		if self.last_sample and len(self.last_sample) == 8:
			target_x, target_y = self.sample_points[point_index]

			self.data_points.append(self.last_sample.copy())
			self.target_points.append([target_x, target_y])
		else:
			print("Warning: invalid sample, skipping")

	def modelFit(self):

		X = np.array(self.data_points)
		y = np.array(self.target_points)

		print("X shape:", X.shape)
		print("y shape:", y.shape)   

		self.model.fit(X, y)
		self.hasCalibrated = True

	def predict(self):
		return self.model.predict([self.last_sample])[0]

	def gaze(self, frame, points):

		image_points = np.array([
			self.relative(points.landmark[4], frame.shape),    # Nose tip
			self.relative(points.landmark[152], frame.shape),  # Chin
			self.relative(points.landmark[263], frame.shape),  # Left eye left corner
			self.relative(points.landmark[33], frame.shape),   # Right eye right corner
			self.relative(points.landmark[287], frame.shape),  # Left Mouth corner
			self.relative(points.landmark[57], frame.shape)    # Right mouth corner
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
			self.relativeT(points.landmark[4], frame.shape),  # Nose tip
			self.relativeT(points.landmark[152], frame.shape),  # Chin
			self.relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
			self.relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
			self.relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
			self.relativeT(points.landmark[57], frame.shape)  # Right mouth corner
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

		left_pupil = self.relative(points.landmark[468], frame.shape)
		right_pupil = self.relative(points.landmark[473], frame.shape)

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

			mean_x = (left_pupil[0] + right_pupil[0]) / 2
			mean_y = (left_pupil[1] + right_pupil[1]) / 2

			dx = right_pupil[0] - left_pupil[0]
			dy = right_pupil[1] - left_pupil[1]

			rx = float(rotation_vector[0][0])
			ry = float(rotation_vector[1][0])
			rz = float(rotation_vector[2][0])

			tx = float(translation_vector[0][0])
			ty = float(translation_vector[1][0])
			tz = float(translation_vector[2][0])

			self.last_sample = [
				mean_x, mean_y,     # average pupil center
				dx, dy,             # eye separation (vergence)
				gaze[0] - left_pupil[0],
				gaze[1] - left_pupil[1],
				right_gaze[0] - right_pupil[0],
				right_gaze[1] - right_pupil[1],
			]

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
		


