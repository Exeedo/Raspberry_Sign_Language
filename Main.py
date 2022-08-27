DEBUG = True
def _print(*args, **kwargs):
	if DEBUG:
		print(*args, **kwargs)

# IMPORTS
_print("Loading Data ...")
import os
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import StandardScaler 
import pickle
import arabic_reshaper

# GLOBALS
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
KEYBOARD2ARABIC = {
	'Q':'ض',
	'W':'ص',
	'E':'ث',
	'R':'ق',
	'T':'ف',
	'Y':'غ',
	'U':'ع',
	'I':'ه',
	'O':'خ',
	'P':'ح',
	'{':'ج',
	'}':'د',
	'A':'ش',
	'S':'س',
	'D':'ي',
	'F':'ب',
	'G':'ل',
	'H':'ا',
	'J':'ت',
	'K':'ن',
	'L':'م',
	':':'ك',
	'"':'ط',
	'Z':'ئ', # unused
	'X':'ء', # unused
	'C':'ؤ', # unused
	'V':'ر',
	'B':'لا',
	'N':'ى', # unused
	'M':'ة',
	'<':'و',
	'>':'ز',
	'?':'ظ',
	'[':'ج',
	']':'د',
	';':'ك',
	"'":'ط',
	',':'و',
	'.':'ز',
	'/':'ظ',
	'`':'ذ',
	'~':'ذ',
}
LABELS = [
	'ain', 'al', 'aleff','bb','dal','dha','dhad','fa','gaaf','ghain','ha','haa','jeem','kaaf','khaa','la','laam',
	'meem','nun','ra','saad','seen','sheen','ta','taa','thaa','thal','toot','waw','ya','yaa','zay'
]
LETTERS_ORDER = ['aleff', 'bb', 'ta', 'thaa', 'jeem', 'haa', 'khaa', 'dal', 'thal', 'ra', 'zay', 'seen', 'sheen', 'saad', 'dhad', 'taa', 'dha', 'ain', 'ghain', 'fa', 'gaaf', 'kaaf', 'laam', 'meem', 'nun', 'ha', 'waw', 'yaa', 'toot', 'al', 'la']
MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
CAMERA = cv2.VideoCapture(0)
SCALER = pickle.load(open('D:/Projects/SignLanguage/Datasets/3/data_scaler.pkl', 'rb'))
MODEL = tf.keras.models.load_model('ver_mp/saved_models/sign_letters_detection_03_points_04')

# HELPER FUNCTIONS
arabic = lambda x: arabic_reshaper.reshape(x)[::-1]
_print("Finished Loading Data ...")

def get_letter(key):
	char = chr(key)
	char_upper = char.upper()
	char_arabic = KEYBOARD2ARABIC.get(char_upper, 'X')
	letter_name = [k for k,v in LETTER_NAME2ARABIC.items() if v[0] == char_arabic] + ['X']
	letter_name = letter_name[0]
	_print(f"Detected {key:3} - {char} - {char_upper} - {arabic(char_arabic)} - {letter_name}")
	return letter_name

def show_letter_image(letter_name, show_hand=True):
	letter_index = LETTERS_ORDER.index(letter_name) + 1
	img_name = os.path.realpath(f'../References/Alphabet/Letter_{letter_index:02}{"_hand" if show_hand else ""}.png')
	x_img_name = os.path.realpath(f'../References/Alphabet/Letter_X.png')
	_print(f"Image name: {img_name}")
	try:
		cv2.imshow("Letter", cv2.imread(img_name))
	except:
		cv2.imshow("Letter", cv2.imread(x_img_name))
		_print("Could not show the image !!!")

def show_word_video(word):
	_print(f"word: {arabic(word)}")

def get_modified_img(img, landmarks):
	MP_DRAWING.draw_landmarks(
		img,
		landmarks,
		MP_HANDS.HAND_CONNECTIONS,
		MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
		MP_DRAWING_STYLES.get_default_hand_connections_style()
	)
	return cv2.flip(img, 1)

def get_points(hand):
	return [
		point.__getattribute__(attr)
			for attr in ['x', 'y', 'z']
				for point in hand.landmark
	]

def process_image(img):
	with MP_HANDS.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.7) as hands:
			ann_img = cv2.flip(img.copy(), 1)
			results = hands.process(cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB))
	img_hands = results.multi_hand_landmarks
	if img_hands is None:
		return None
	modified_img = get_modified_img(cv2.flip(img.copy(), 1), img_hands[0])
	# cv2.imshow('Modified-Image', modified_img)
	return get_points(img_hands[0])

def predict(cv2_img):
	points = process_image(cv2_img)
	if not points:
		_print("Could not detect a hand in this image !")
		return
	preprocessed_points = SCALER.transform(np.array(points).reshape(1, -1));
	predictions = MODEL.predict(preprocessed_points);
	score = tf.nn.softmax(predictions[0]);
	label = LABELS[np.argmax(score)];
	ar_letter = LETTER_NAME2ARABIC[label][1];
	show_letter_image(label, show_hand=False)
	_print(f"This image most likely belongs to '{arabic(ar_letter)}' "
			f"with a {100 * np.max(score):.2f} percent confidence.")

# MAIN FUNCTIONS
def DisplayMode():
	_print("Entering Display mode")
	_print("Press any letter to show it")
	_print("Press Enter or Space to show the resulting word")
	_print("Press Escape to exit this mode")
	cv2.destroyAllWindows()
	cv2.namedWindow("Letter")
	word = ''
	while True:
		key = cv2.waitKey(20)
		if key == 32: # space
			_print("Space bar detected ...")
			show_word_video(word)
			word = ''
			continue
		if key == 13: # enter
			_print("Enter detected ...")
			show_word_video(word)
			word = ''
			continue
		if key == 8: # backspace
			_print("Backspace detected ...")
			word = word[:-1]
			continue
		if key == 27: # escape
			_print("Escape detected ...")
			break
		if key != -1:
			letter_name = get_letter(key)
			if letter_name != 'X':
				word += LETTER_NAME2ARABIC[letter_name][0]
				show_letter_image(letter_name)
	_print("Exiting Display mode")

def DetectionMode():
	_print("Entering Detection mode")
	_print("Point the camera to your hand")
	_print("Press Enter or Space to run the detection")
	_print("Press Escape to exit this mode")
	cv2.destroyAllWindows()
	cv2.namedWindow("Camera-Live")
	while True:
		_, frame = CAMERA.read()
		cv2.imshow("Camera-Live", frame)
		key = cv2.waitKey(20)
		if key == 13: # enter
			_print("Enter detected ...")
			predict(frame)
		if key == 27: # escape
			_print("Escape detected ...")
			break
	_print("Exiting Detection mode")

def ModeChoose():
	all_modes = [ModeChoose, DisplayMode, DetectionMode]
	welcome_img = cv2.imread('Welcome.png')
	while True:
		cv2.destroyAllWindows()
		_print("Entering Mode Choose")
		for i, m in enumerate(all_modes):
			if not i: continue
			_print(f"Press {i} to choose {m.__name__} Mode")
		_print("Press Escape to exit the program")
		cv2.namedWindow("Start")
		cv2.imshow("Start", welcome_img)
		while True:
			mode_number = 0
			key = cv2.waitKey(20)
			if key == -1:
				continue
			if key == 27: # escape
				_print("Escape detected ...")
				return
			if chr(key).isdigit() and int(chr(key)) in range(len(all_modes)):
				mode_number = int(chr(key))
			if mode_number != 0: 
				break
		all_modes[mode_number]()

if __name__ == '__main__':
	print("Program started")
	ModeChoose()
	CAMERA.release()
	cv2.destroyAllWindows()
	print("Program ended")
