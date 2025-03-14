from flask import Flask, render_template, Response, redirect, url_for, jsonify, send_from_directory, request
import os
import mediapipe as mp
import cv2
import math
import datetime
import numpy as np
import threading
import re

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
        self.history = []
        self.history_limit = 10
        self.last_gesture_time = datetime.datetime.now()
        self.gesture_cooldown = 0.5
        self.current_gesture = None
        
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def is_finger_extended(self, hand_landmarks, finger_tip_idx, finger_pip_idx, threshold=0.05):
        """Check if a finger is extended"""
        return hand_landmarks.landmark[finger_tip_idx].y < hand_landmarks.landmark[finger_pip_idx].y - threshold
    
    def is_finger_folded(self, hand_landmarks, finger_tip_idx, finger_pip_idx, threshold=0.05):
        """Check if a finger is folded"""
        return hand_landmarks.landmark[finger_tip_idx].y > hand_landmarks.landmark[finger_pip_idx].y + threshold
    
    def get_finger_states(self, hand_landmarks):
        """Get the state of all fingers"""
        states = {
            'thumb': self.is_finger_extended(hand_landmarks, 4, 3),
            'index': self.is_finger_extended(hand_landmarks, 8, 6),
            'middle': self.is_finger_extended(hand_landmarks, 12, 10),
            'ring': self.is_finger_extended(hand_landmarks, 16, 14),
            'pinky': self.is_finger_extended(hand_landmarks, 20, 18)
        }
        return states
    
    def detect_gesture(self, image):
        results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        new_gesture = None
        
        if results.multi_hand_landmarks:
            current_time = datetime.datetime.now()
            time_diff = (current_time - self.last_gesture_time).total_seconds()
            
            if time_diff >= self.gesture_cooldown:
                hand_landmarks = results.multi_hand_landmarks[0]
                new_gesture = self.get_gesture(hand_landmarks)
                
                if new_gesture != self.current_gesture:
                    if new_gesture is not None:
                        timestamp = current_time.strftime("%H:%M:%S")
                        self.history.append({
                            'gesture': new_gesture,
                            'timestamp': timestamp
                        })
                        if len(self.history) > self.history_limit:
                            self.history.pop(0)
                    
                    self.current_gesture = new_gesture
                    self.last_gesture_time = current_time
        
        return results
    
    def get_gesture(self, hand_landmarks):
        # Get landmarks for easier reference
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Get finger states
        states = self.get_finger_states(hand_landmarks)
        
        # Live Long (Vulcan) Sign
        if (states['index'] and states['middle'] and 
            states['ring'] and states['pinky'] and
            abs(middle_tip.x - ring_tip.x) > 0.04 and
            abs(index_tip.y - middle_tip.y) < 0.03 and
            abs(ring_tip.y - pinky_tip.y) < 0.03):
            return "Live Long 🖖"
        
        # Fist
        # if (all(not state for state in states.values()) and
        #     not states['thumb']):
        #     return "Fist ✊"
        
        # Point Right
        if (states['index'] and not states['middle'] and
            not states['ring'] and not states['pinky'] and
            index_tip.x > hand_landmarks.landmark[5].x):
            return "Point Right 👉"
        
        # Point Left
        if (states['index'] and not states['middle'] and
            not states['ring'] and not states['pinky'] and
            index_tip.x < hand_landmarks.landmark[5].x):
            return "Point Left 👈"
        
        # Point Up
        if (states['index'] and not states['middle'] and
            not states['ring'] and not states['pinky'] and
            index_tip.y < hand_landmarks.landmark[5].y - 0.1):
            return "Point Up 👆"
        
        # Point Down
        if (states['index'] and not states['middle'] and
            not states['ring'] and not states['pinky'] and
            index_tip.y > hand_landmarks.landmark[5].y + 0.1):
            return "Point Down 👇"
        
        # Heart
        thumb_index_dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        if (thumb_index_dist < 0.1 and
            not states['middle'] and not states['ring'] and
            not states['pinky'] and
            thumb_tip.x < index_tip.x and
            abs(thumb_tip.y - index_tip.y) < 0.05):
            return "Heart ❤️"
        
        # Peace
        if (states['index'] and states['middle'] and
            not states['ring'] and not states['pinky'] and
            abs(index_tip.x - middle_tip.x) < 0.08 and
            abs(index_tip.y - middle_tip.y) < 0.08):
            return "Peace ✌️"
        
        # Rock On
        if (states['index'] and not states['middle'] and
            not states['ring'] and states['pinky']):
            return "Rock On 🤘"
        
        # OK Sign
        if (abs(thumb_tip.x - index_tip.x) < 0.05 and
            abs(thumb_tip.y - index_tip.y) < 0.05 and
            states['middle'] and states['ring'] and states['pinky']):
            return "OK 👌"
        
        # Thumbs Up
        if (states['thumb'] and thumb_tip.y < hand_landmarks.landmark[2].y and
            not any(states[finger] for finger in ['index', 'middle', 'ring', 'pinky'])):
            return "Thumbs Up 👍"
        
        # Thumbs Down
        if (states['thumb'] and thumb_tip.y > hand_landmarks.landmark[2].y and
            not any(states[finger] for finger in ['index', 'middle', 'ring', 'pinky'])):
            return "Thumbs Down 👎"
        
        # Stop/High Five
        if (all(states[finger] for finger in ['index', 'middle', 'ring', 'pinky']) and
            abs(index_tip.y - pinky_tip.y) < 0.1):
            return "Stop/High Five ✋"
        
        # Wave
        if (all(states[finger] for finger in ['index', 'middle', 'ring', 'pinky']) and
            abs(index_tip.x - pinky_tip.x) > 0.15):
            return "Wave 👋"

        # Thank You
        if (all(not states[finger] for finger in ['index', 'middle', 'ring', 'pinky']) and
            states['thumb'] and
            thumb_tip.x < hand_landmarks.landmark[5].x):
            return "Thank You 🙏"
        
        return None

    def release(self):
        self.hands.close()

# Initialize the converter and capture with error handling
def init_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)  # Try another camera index
    if not camera.isOpened():
        raise RuntimeError("No camera found")
    return camera

try:
    sign_lang_conv = SignLanguageConverter()
    camera = init_camera()
except Exception as e:
    print(f"Error initializing camera: {e}")

def remove_emoji(text):
    """Remove emoji characters from text using regex pattern"""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs (covers 🤘)
        u"\U00002600-\U000026FF"  # misc symbols (covers ❤️)
        u"\U00002700-\U000027BF"  # dingbats (covers ✌️)
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text).strip()

def generate_frames():
    global camera
    
    while True:
        try:
            success, frame = camera.read()
            if not success:
                camera.release()
                camera = init_camera()
                continue
            
            results = sign_lang_conv.detect_gesture(frame)
            gesture = sign_lang_conv.current_gesture
            
            if gesture:
                # Remove emoji from gesture text
                clean_gesture = remove_emoji(gesture)
                cv2.putText(frame, clean_gesture, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            try:
                camera.release()
            except:
                pass
            camera = init_camera()
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html', history=sign_lang_conv.history)

@app.route('/get_history')
def get_history():
    return jsonify({'history': sign_lang_conv.history})

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), 
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    file = request.files['frame']
    
    # Convert the received image file to a numpy array
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the frame with MediaPipe
    results = sign_lang_conv.detect_gesture(image)
    gesture = sign_lang_conv.current_gesture
    
    # Return the detected gesture
    return jsonify({
        'gesture': gesture if gesture else None,
        'added_to_history': len(sign_lang_conv.history) > 0 and 
                           sign_lang_conv.history[-1]['gesture'] == gesture
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    sign_lang_conv.history.clear()
    return redirect(url_for('detect'))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.svg', mimetype='image/vnd.microsoft.icon')

@app.route('/save_history')
def save_history():
    history_text = "\n".join(
        [f"{entry['timestamp']}: {entry['gesture']}" for entry in sign_lang_conv.history]
    )
    return Response(
        history_text,
        mimetype="text/plain",
        headers={"Content-disposition": "attachment; filename=gesture_history.txt"}
    )

if __name__ == '__main__':
    app.run(debug=True)