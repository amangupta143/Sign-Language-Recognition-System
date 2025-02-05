from flask import Flask, render_template, Response
import mediapipe as mp
import cv2
import math
import datetime
import pyttsx3

app = Flask(__name__)

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class SignLanguageConverter:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.current_gesture = None
    
    def detect_gesture(self, image):
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.current_gesture = self.get_gesture(hand_landmarks)
        return results
    
    def get_gesture(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]
        ring_finger_tip = hand_landmarks.landmark[16]
        little_finger_tip = hand_landmarks.landmark[20]

        # Check if hand is in OK gesture
        if thumb_tip.y < index_finger_tip.y < middle_finger_tip.y < ring_finger_tip.y < little_finger_tip.y:
            return "Okay"

        # Check if hand is in Dislike gesture
        elif thumb_tip.y > index_finger_tip.y > middle_finger_tip.y > ring_finger_tip.y > little_finger_tip.y:
            return "I dislike It"

        # Check if hand is in Victory gesture
        elif index_finger_tip.y < middle_finger_tip.y and abs(index_finger_tip.x - middle_finger_tip.x) < 0.2:
            return "We Won! Victory"

        # Check if hand is in Stop gesture
        elif thumb_tip.x < index_finger_tip.x < middle_finger_tip.x:
            if (hand_landmarks.landmark[2].x < hand_landmarks.landmark[5].x) and \
               (hand_landmarks.landmark[3].x < hand_landmarks.landmark[5].x) and \
               (hand_landmarks.landmark[4].x < hand_landmarks.landmark[5].x):
                return "STOP! Dont Move."
            
        # Check if hand is in Point gesture
        wrist = hand_landmarks.landmark[0]
        index_finger = (index_finger_tip.x, index_finger_tip.y, index_finger_tip.z)
        wrist_coords = (wrist.x, wrist.y, wrist.z)
        vector = (index_finger[0] - wrist_coords[0], 
                 index_finger[1] - wrist_coords[1], 
                 index_finger[2] - wrist_coords[2])
        vector_len = math.sqrt(sum(x*x for x in vector))
        if vector_len != 0:
            vector_unit = tuple(x/vector_len for x in vector)
            reference_vector = (0, 0, -1)
            dot_product = sum(a*b for a, b in zip(vector_unit, reference_vector))
            angle = math.acos(max(min(dot_product, 1), -1)) * 180 / math.pi
            if 20 < angle < 80:
                return "Hey You!!"
        
        return None

    def release(self):
        self.hands.close()

# Initialize the converter and capture
sign_lang_conv = SignLanguageConverter()
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process the frame
        results = sign_lang_conv.detect_gesture(frame)
        gesture = sign_lang_conv.current_gesture
        
        # Draw gesture text
        if gesture:
            cv2.putText(frame, gesture, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)