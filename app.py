from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import time
import threading
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import os
from collections import deque

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

label_translation = {
    "thik hee": {"hi": "ठीक है", "en": "Okay"},
    "namaste": {"hi": "नमस्ते", "en": "Hello"},
    "haan": {"hi": "हाँ", "en": "Yes"},
    "nahi": {"hi": "नहीं", "en": "No"},
    "achha hee": {"hi": "अच्छा है", "en": "Good"},
    "khaana": {"hi": "खाना", "en": "Eat"},
    "madad": {"hi": "मदद", "en": "Help"},
    "paani": {"hi": "पानी", "en": "Water"},
    "rooko": {"hi": "रुको", "en": "Wait"},
    "tum": {"hi": "आप", "en": "You"},
}

system_messages = {
    "no_sign": {"hi": "कोई संकेत मिल नहीं रहा है!", "en": "No Sign Recognized!"},
    "unknown": {"hi": "अज्ञात संकेत", "en": "Unknown Sign"},
}

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    label_map = model_dict['label_map']
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model, label_map = None, {}

last_spoken_text = None
last_detection_time = 0
prediction_queue = deque(maxlen=15)
prediction_delay = 2 
no_sign_delay = 10
hold_sign_time = 2
current_language = "hi"
speech_lock = threading.Lock()
last_predicted_text = None
last_no_sign_time = time.time()
no_sign_spoken = False
last_sign_detected_time = time.time() 
last_prediction_time = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('set_language')
def set_language(data):
    global current_language
    current_language = data.get("language", "hi")
    print(f"Language set to: {current_language}")

def text_to_speech(text, lang="hi"):
    global last_spoken_text

    if text.strip() and text != last_spoken_text:
        last_spoken_text = text
        print(f"Speaking: {text}")

        with speech_lock:
            try:
                audio_file = "temp_audio.mp3"
                tts = gTTS(text=text, lang=lang)
                tts.save(audio_file)

                data, samplerate = sf.read(audio_file)
                sd.play(data, samplerate)
                sd.wait()

                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e:
                print("Speech Error:", e)

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Cannot access webcam!")
        return

    print("Camera Accessed Successfully!")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    global last_detection_time, last_predicted_text, last_no_sign_time, no_sign_spoken, last_sign_detected_time, last_prediction_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame!")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()
        detected_sign = None

        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            try:
                prediction = model.predict([np.asarray(data_aux)])[0]
                predicted_text_hindi = label_map.get(prediction, "Unknown")

                if predicted_text_hindi in label_translation:
                    translated_text = label_translation[predicted_text_hindi].get(current_language, "Unknown")
                else:
                    translated_text = system_messages["unknown"][current_language]

                prediction_queue.append(translated_text)

                if prediction_queue.count(translated_text) >= 4:
                    if (current_time - last_detection_time) >= hold_sign_time:
                        if (current_time - last_prediction_time) >= prediction_delay:
                            if translated_text != last_predicted_text:
                                print(f"Confirmed Sign: {translated_text}")
                                socketio.emit('prediction', {'text': translated_text})
                                threading.Thread(target=text_to_speech, args=(translated_text, current_language), daemon=True).start()
                                last_predicted_text = translated_text
                                last_prediction_time = current_time 

                        last_detection_time = current_time
                        last_sign_detected_time = current_time

                detected_sign = translated_text
                no_sign_spoken = False

            except Exception as e:
                print("Prediction Error:", e)

        if detected_sign is None:
            if (current_time - last_sign_detected_time) >= no_sign_delay:
                if not no_sign_spoken:
                    print("No Sign Recognized!")
                    socketio.emit('prediction', {'text': system_messages["no_sign"][current_language]})
                    threading.Thread(target=text_to_speech, args=(system_messages["no_sign"][current_language], current_language), daemon=True).start()
                    last_no_sign_time = current_time
                    no_sign_spoken = True

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print("Error encoding frame:", e)
            break

        cv2.waitKey(1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Server starting...")
    socketio.run(app, debug=False, host="127.0.0.1", port=5000, allow_unsafe_werkzeug=True)
