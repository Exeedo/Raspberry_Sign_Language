import os
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import arabic_reshaper
from sklearn.preprocessing import StandardScaler
import pickle
import warnings

warnings.filterwarnings('ignore')

LETTER_NAME2ARABIC = {
	'aleff'	: ('ا', 'ألف'),
	'bb'	: ('ب', 'باء'),
	'ta'	: ('ت', 'تاء'),
	'thaa'	: ('ث', 'ثاء'),
	'jeem'	: ('ج', 'جيم'),
	'haa'	: ('ح', 'حاء'),
	'khaa'	: ('خ', 'خاء'),
	'dal'	: ('د', 'دال'),
	'thal'	: ('ذ', 'ذال'),
	'ra'	: ('ر', 'راء'),
	'zay'	: ('ز', 'زاي'),
	'seen'	: ('س', 'سين'),
	'sheen'	: ('ش', 'شين'),
	'saad'	: ('ص', 'صاد'),
	'dhad'	: ('ض', 'ضاد'),
	'taa'	: ('ط', 'طاء'),
	'dha'	: ('ظ', 'ظاء'),
	'ain'	: ('ع', 'عين'),
	'ghain'	: ('غ', 'غين'),
	'fa'	: ('ف', 'فاء'),
	'gaaf'	: ('ق', 'قاف'),
	'kaaf'	: ('ك', 'كاف'),
	'laam'	: ('ل', 'لام'),
	'meem'	: ('م', 'ميم'),
	'nun'	: ('ن', 'نون'),
	'ha'	: ('ه', 'هاء'),
	'waw'	: ('و', 'واو'),
	'yaa'	: ('ي', 'ياء'),
	'toot'	: ('ة', 'تاء مربوطة'),
	'al'	: ('ال', 'ال التعريف'),
	'la'	: ('لا', 'لا'),
	'ya'	: ('يا', 'يا'),
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
arabic = lambda x: arabic_reshaper.reshape(x)[::-1]

model = tf.keras.models.load_model('saved_models/sign_letters_detection_03_points_04')
# model.summary()
labels = [
	'ain', 'al', 'aleff','bb','dal','dha','dhad','fa','gaaf','ghain','ha','haa','jeem','kaaf','khaa','la','laam',
	'meem','nun','ra','saad','seen','sheen','ta','taa','thaa','thal','toot','waw','ya','yaa','zay'
	]

scaler = pickle.load(open('D:/Projects/SignLanguage/Datasets/3/data_scaler.pkl', 'rb'))

def get_modified_img(img, landmarks):
	mp_drawing.draw_landmarks(
		img,
		landmarks,
		mp_hands.HAND_CONNECTIONS,
		mp_drawing_styles.get_default_hand_landmarks_style(),
		mp_drawing_styles.get_default_hand_connections_style()
	)
	return cv2.flip(img, 1)

def get_points(hand):
	return [
		point.__getattribute__(attr)
			for attr in ['x', 'y', 'z']
				for point in hand.landmark
	]

def process_image(img):
	with mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.7) as hands:
			ann_img = cv2.flip(img.copy(), 1)
			results = hands.process(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB))
	img_hands = results.multi_hand_landmarks
	if img_hands is None:
		return None
	modified_img = get_modified_img(cv2.flip(img.copy(), 1), img_hands[0])
	cv2.imshow('Modified-Image', modified_img)
	return get_points(img_hands[0])

def predict(cv2_img):
	# cv2.imshow("Camera-Saved", cv2_img)
	points = process_image(cv2_img)
	if not points:
		print("Could not detect a hand in this image !")
		return
	# breakpoint()
	preprocessed_points = scaler.transform(np.array(points).reshape(1, -1));
	predictions = model.predict(preprocessed_points);
	score = tf.nn.softmax(predictions[0]);
	label = labels[np.argmax(score)];
	ar_letter = LETTER_NAME2ARABIC[label][1];
	print(f"This image most likely belongs to '{arabic(ar_letter)}' "
			f"with a {100 * np.max(score):.2f} percent confidence.")

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera-Live")
while True:
	_, frame = cam.read()
	cv2.imshow("Camera-Live", frame)
	key = cv2.waitKey(20)
	if key == 13: # enter
		# print("Enter detected ...")
		predict(frame)
	if key == 27: # escape
		print("Escape detected ...")
		break
cam.release()
cv2.destroyAllWindows()
print("Program ended.")