DEBUG = True
def _print(*args, **kwargs):
	if DEBUG:
		print(*args, **kwargs)

# IMPORTS
_print("Loading Data ...")
import os
from time import time
import tensorflow as tf
import cv2
from screeninfo import get_monitors
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
WORDS2VIDEO_NUMBER = {
	'كبير': 1,
	'صغير': 2,
	'كثير': 3,
	'قليل': 4,
	'قديم': 5,
	'جديد': 6,
	'خربان': 7,
	'مكسور': 8,
	'بعيد': 9,
	'قريب': 10,
	'مفتوح': 11,
	'مغلق': 12,
	'سهل': 13,
	'صعب': 14,
	'قوي': 15,
	'ضعيف': 16,
	'ملان': 17,
	'فارغ': 18,
	'نظيف': 19,
	'قذر': 20,
	'سريع': 21,
	'بطيء': 22,
	'سمين': 23,
	'نحيف': 24,
	'الشيء نفسه': 25,
	'مختلف': 26,
	'سيء السلوك': 27,
	'مؤدب': 28,
	'فوضى': 29,
	'مرتب': 30,
	'جيد': 31,
	'احسن': 32,
	'سيء': 33,
	'ذكي': 34,
	'عنيد': 35,
	'مجنون': 36,
	'عصبي': 37,
	'كسلان': 38,
	'خطر': 39,
	'هام': 40,
	'مهم': 40,
	'صباح': 41,
	'ظهر': 42,
	'عصر': 43,
	'مغرب': 44,
	'ليل': 45,
	'السبت': 46,
	'الاحد': 47,
	'الاثنين': 48,
	'الثلاثاء': 49,
	'الاربعاء': 50,
	'الخميس': 51,
	'الجمعة': 52,
	'يوم': 53,
	'اسبوع': 54,
	'شهر': 55,
	'سنة': 56,
	'زمان': 57,
	'اليوم': 58,
	'امس': 59,
	'غدا': 60,
	'بكرة': 60,
	'مستقبل': 61,
	'دائما': 62,
	'على طول': 63,
	'احيانا': 64,
	'ابدا': 65,
	'وقت': 66,
	'دقيقة': 67,
	'ساعة': 68,
	'مبكر': 69,
	'متاخر': 70,
	'مدة طويلة': 71,
	'متى': 72,
	'باص': 73,
	'بحر': 74,
	'برد': 75,
	'حار': 76,
	'ربيع': 77,
	'شارع': 78,
	'شتاء': 79,
	'صيف': 80,
	'طائر': 81,
	'شرطي': 84,
	'شمس': 85,
	'قمر': 86,
	'مطر': 87,
	'زهرة': 88,
	'شجرة': 89,
	'جبل': 90,
	'الاردن': 91,
	'الزرقاء': 92,
	'الكرك': 93,
	'عمان': 94,
	'معان': 95,
	'الرمثا': 96,
	'اربد': 97,
	'عجلون': 98,
	'السلط': 99,
	'الغور': 100,
	'العقبة': 101,
	'المفرق': 102,
	'مدرسة': 103,
	'محل': 104,
	'دكان': 104,
	'جامع': 105,
	'مطعم': 106,
	'مستشفى': 107,
	'نزهة': 108,
	'جار': 109,
	'سيارة': 110,
	'شكرا': 111,
	'مرحبا': 111,
	'كيف حالك': 112,
	'بخير': 113,
	'وداعا': 114,
	'تصبح على خير': 115,
	'ارجوك': 116,
	'ممنوع': 117,
	'عادي': 118,
	'عيب': 119,
	'يوجد': 120,
	'لا يوجد': 121,
	'بجد': 122,
	'مرة ثانية': 123,
	'مع': 124,
	'مع بعض': 125,
	'الكل': 126,
	'هنا': 127,
	'هناك': 128,
	'اسكت': 129,
	'نعم': 130,
	'لا': 131,
	'الاسم': 132,
	'ماذا': 133,
	'لماذا': 134,
	'كيف': 135,
	'من': 136,
	'اين': 137,
	'ما السبب': 138,
	'كم': 139,
	'يمشي': 140,
	'يركض': 141,
	'يقف': 142,
	'يجلس': 143,
	'ياكل': 144,
	'يلعب': 145,
	'ينام': 146,
	'يدرس': 147,
	'يرسم': 148,
	'يساعد': 149,
	'يذهب': 150,
	'ياتي': 151,
	'يعرف': 152,
	'لا يعرف': 153,
	'يتذكر': 154,
	'مبسوط': 155,
	'متلهف': 156,
	'زعلان': 157,
	'حزين': 158,
	'مريض': 159,
	'تعبان': 160,
	'محرج': 161,
	'خائف': 162,
	'متفاجئ': 163,
	'احب': 164,
	'لا احب': 165,
	'يبكي': 166,
	'يضحك': 167,
	'اريد': 168,
	'يشعر بالملل': 169,
}
LETTERS_ORDER = ['X', 'aleff', 'bb', 'ta', 'thaa', 'jeem', 'haa', 'khaa', 'dal', 'thal', 'ra', 'zay', 'seen', 'sheen', 'saad', 'dhad', 'taa', 'dha', 'ain', 'ghain', 'fa', 'gaaf', 'kaaf', 'laam', 'meem', 'nun', 'ha', 'waw', 'yaa', 'toot', 'al', 'la']
MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles
CAMERA = cv2.VideoCapture(0)
SCALER = pickle.load(open('D:/Projects/SignLanguage/Datasets/3/data_scaler.pkl', 'rb'))
MODEL = tf.keras.models.load_model('ver_mp/saved_models/sign_letters_detection_03_points_04')

# HELPER FUNCTIONS
arabic = lambda x: arabic_reshaper.reshape(x)[::-1]
_print("Finished Loading Data ...")

def show_centered_image(window_name, img):
	screen_center = (MONITOR.width // 2, MONITOR.height // 2)
	img_center = (img.shape[1] // 2, img.shape[0] // 2)
	img_pos = (screen_center[0] - img_center[0], screen_center[1] - img_center[1])
	show_image_at(window_name, img, img_pos)

def show_image_top_mid(window_name, img):
	screen_center = (MONITOR.width // 2, MONITOR.height // 2)
	img_center = (img.shape[1] // 2, img.shape[0] // 2)
	img_pos = (screen_center[0] - img_center[0], 0)
	show_image_at(window_name, img, img_pos)

def show_image_top_left(window_name, img):
	img_pos = (0, 0)
	show_image_at(window_name, img, img_pos)

def show_image_top_right(window_name, img):
	img_pos = (MONITOR.width - img.shape[1], 0)
	show_image_at(window_name, img, img_pos)

def show_image_bottom_left(window_name, img):
	img_pos = (0, MONITOR.height - img.shape[0])
	show_image_at(window_name, img, img_pos)

def show_image_bottom_right(window_name, img):
	img_pos = (MONITOR.width - img.shape[1], MONITOR.height - img.shape[0])
	show_image_at(window_name, img, img_pos)

def show_image_at(window_name, img, img_pos):
	print(f"Showing {window_name} at {img_pos}")
	cv2.moveWindow(window_name, *img_pos)
	cv2.imshow(window_name, img)

def get_letter(key):
	char = chr(key)
	char_upper = char.upper()
	char_arabic = KEYBOARD2ARABIC.get(char_upper, 'X')
	letter_name = [k for k,v in LETTER_NAME2ARABIC.items() if v[0] == char_arabic] + ['X']
	letter_name = letter_name[0]
	_print(f"Detected {key:3} - {char} - {char_upper} - {arabic(char_arabic)} - {letter_name}")
	return letter_name, char_arabic

def show_letter_image(letter_name, show_hand=True):
	letter_index = LETTERS_ORDER.index(letter_name)
	img_name = os.path.realpath(f'../References/Alphabet/Letter_{letter_index:02}{"_hand" if show_hand else ""}.png')
	_print(f"Image name: {img_name}")
	try:
		if show_hand:
			show_image_top_left("Letter", cv2.imread(img_name))
		else:
			show_image_top_right("Letter", cv2.imread(img_name))
	except:
		_print("Could not show the image !!!")

def show_word_video(word):
	_print(f"word: {arabic(word)}")
	if word not in WORDS2VIDEO_NUMBER:
		_print(f"Cannot find video for {arabic(word)}")
		return
	video_number = WORDS2VIDEO_NUMBER[word]
	video_file = os.path.realpath(f'../References/Words/{video_number:03}.MP4')
	if not os.path.exists(video_file):
		_print(f"Video for {arabic(word)} was not added yet")
		return
	cv2.namedWindow("Word", cv2.WINDOW_NORMAL)
	cv2.setWindowProperty("Word", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	vid = cv2.VideoCapture(video_file)
	fps = vid.get(cv2.CAP_PROP_FPS) # 50
	frame_ms = 1000 / fps # 20 ms/frame
	now = time()
	start = now
	while vid.isOpened():
		while time() - now < frame_ms/1000:
			pass
		now = time()
		ret, frame = vid.read()
		if not ret:
			break
		cv2.imshow("Word", frame)
		cv2.waitKey(1)
	_print(f"Elapsed {time() - start} seconds")
	cv2.destroyWindow("Word")
	vid.release()

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
	cv2.namedWindow("Letter")
	show_letter_image(label, show_hand=False)
	_print(f"This image most likely belongs to '{arabic(ar_letter)}' "
			f"with a {100 * np.max(score):.2f} percent confidence.")

# MAIN FUNCTIONS
def DisplayMode():
	_print("Entering Display mode")
	_print("Press any letter to show it")
	_print("Press Enter to show the resulting word")
	_print("Press Escape to exit this mode")
	cv2.destroyAllWindows()
	cv2.namedWindow("Letter")
	word = ''
	reset_word = False
	while True:
		key = cv2.waitKey(20)
		if key == 32: # space
			_print("Space bar detected ...")
			word += ' '
			continue
		if key == 13: # enter
			_print("Enter detected ...")
			show_word_video(word)
			reset_word = True
			continue
		if key == 8: # backspace
			_print("Backspace detected ...")
			word = word[:-1]
			continue
		if key == 27: # escape
			_print("Escape detected ...")
			break
		if key != -1:
			letter_name, char_arabic = get_letter(key)
			if letter_name != 'X':
				if reset_word:
					word = ''
					reset_word = False
				word += LETTER_NAME2ARABIC[letter_name][0]
				show_letter_image(letter_name)
			else:
				if char_arabic != 'X':
					word += char_arabic
				show_letter_image(letter_name)
	_print("Exiting Display mode")

def DetectionMode():
	_print("Entering Detection mode")
	_print("Point the camera to your hand")
	_print("Press Enter or Space to run the detection")
	_print("Press Escape to exit this mode")
	cv2.destroyAllWindows()
	cv2.namedWindow("Camera-Live")
	_, frame = CAMERA.read()
	show_image_top_left("Camera-Live", frame)
	while True:
		_, frame = CAMERA.read()
		cv2.imshow("Camera-Live", frame)
		key = cv2.waitKey(20)
		if key == 32: # space
			_print("Space bar detected ...")
			predict(frame)
		if key == 13: # enter
			_print("Enter detected ...")
			predict(frame)
		if key == 27: # escape
			_print("Escape detected ...")
			break
	cv2.destroyWindow("Camera-Live")
	cv2.destroyWindow("Letter")
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
		show_image_top_mid("Start", welcome_img)
		while True:
			mode_number = 0
			key = cv2.waitKey(0)
			if key == 27: # escape
				_print("Escape detected ...")
				return
			if chr(key).isdigit() and int(chr(key)) in range(len(all_modes)):
				mode_number = int(chr(key))
			if mode_number != 0: 
				break
		all_modes[mode_number]()

if __name__ == '__main__':
	MONITOR = get_monitors()[0]
	print("Program started")
	ModeChoose()
	CAMERA.release()
	cv2.destroyAllWindows()
	print("Program ended")
