import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

get_hand_type = lambda res, idx : res.multi_handedness[idx].classification[0].label

def process_image(img_path):
	img = cv2.imread(img_path)
	with mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.7) as hands:
			ann_img = cv2.flip(img.copy(), 1)
			results = hands.process(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB))
	img_hands = results.multi_hand_landmarks
	if img_hands is None:
		return None, None
	if len(img_hands) == 1:
		breakpoint()
		save_img(get_modified_img(ann_img, img_hands[0]), img_path)
		return get_points(img_hands[0]), get_hand_type(results, 0)
	for i in range(2):
		modified_img = get_modified_img(cv2.flip(img.copy(), 1), img_hands[i])
		if check_img(modified_img):
			cv2.destroyAllWindows()
			save_img(modified_img, img_path)
			return get_points(img_hands[i]), get_hand_type(results, i)
	return None, None

def get_points(hand):
	return [
		point.__getattribute__(attr)
			for attr in ['x', 'y', 'z']
				for point in hand.landmark
	]

def get_modified_img(img, landmarks):
	mp_drawing.draw_landmarks(
		img,
		landmarks,
		mp_hands.HAND_CONNECTIONS,
		mp_drawing_styles.get_default_hand_landmarks_style(),
		mp_drawing_styles.get_default_hand_connections_style()
	)
	return cv2.flip(img, 1)

def save_img(img, img_path):
	file_path = img_path.replace('images', 'modified_images')
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	cv2.imwrite(file_path, img)

def check_img(img):
	while True:
		cv2.imshow('Image', img)
		key = cv2.waitKey(0)
		if chr(key).upper() == 'Y':
			return True
		if chr(key).upper() == 'N':
			return False

def process_dir(data_dir):
	subfolders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
	for subfolder in subfolders:
		all_points_float = []
		all_info_str = []
		subfolder_path = os.path.join(data_dir, subfolder, 'images')
		for folder in os.listdir(subfolder_path):
			folder_path = os.path.join(subfolder_path, folder)
			for img in os.listdir(folder_path):
				img_path = os.path.join(folder_path, img)
				points, hand_type = process_image(img_path)
				if points:
					all_points_float.append(points)
					all_info_str.append((
						img_path.replace('images', 'modified_images'),
						folder,
						hand_type
					))
		points_matrix = np.array(all_points_float, dtype=float)
		points_df = pd.DataFrame(
			points_matrix,
			columns=[f'p{i+1:02}{c}' for i in range(21) for c in ['x', 'y', 'z']],
			dtype=float
		)
		img_df = pd.DataFrame(
			all_info_str,
			columns=['img_path', 'label', 'type'],
			dtype=str
		)
		main_df = pd.concat([img_df['img_path'], points_df, img_df['type'], img_df['label']], axis=1)
		main_df.to_csv(os.path.join(subfolder_path, "all_points_with_handedness.csv"))

process_dir(os.path.realpath('../Datasets/3/'))
process_dir(os.path.realpath('../Datasets/4/'))
