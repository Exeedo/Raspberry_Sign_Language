import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_image(image):
	with mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.7) as hands:
		annotated_image = cv2.flip(image.copy(), 1)
		results = hands.process(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
	if not results.multi_hand_landmarks:
		return None
	for hand_landmarks in results.multi_hand_landmarks:
		mp_drawing.draw_landmarks(
			annotated_image,
			hand_landmarks,
			mp_hands.HAND_CONNECTIONS,
			mp_drawing_styles.get_default_hand_landmarks_style(),
			mp_drawing_styles.get_default_hand_connections_style()
		)
	cv2.imshow("Camera-Live", cv2.flip(annotated_image, 1))

if __name__ == '__main__':
	cv2.namedWindow("Camera-Live")
	camera = cv2.VideoCapture(0)
	while True:
		_, frame = camera.read()
		cv2.imshow("Camera-Live", frame)
		process_image(frame)
		key = cv2.waitKey(50)
		if key == 27: # escape
			print("Escape detected ...")
			break
	camera.release()
	cv2.destroyAllWindows()
