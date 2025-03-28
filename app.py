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

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

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
steady_sign = None
prediction_delay = 2

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

def text_to_speech(text, lang='hi'):
    global last_spoken_text

    if text.strip() and text != last_spoken_text:
        last_spoken_text = text
        print(f"Speaking: {text}")

        try:
            tts = gTTS(text=text, lang=lang)
            tts.save("temp_audio.mp3")

            data, samplerate = sf.read("temp_audio.mp3")
            sd.play(data, samplerate, blocking=False)
            os.remove("temp_audio.mp3")

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
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    global last_detection_time, steady_sign

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame!")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()

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
                predicted_text = label_map.get(prediction, "Unknown")

                if predicted_text != steady_sign:
                    steady_sign = predicted_text
                    last_detection_time = current_time

                elif (current_time - last_detection_time) >= prediction_delay:
                    print(f"Confirmed Sign: {predicted_text}")
                    socketio.emit('prediction', {'text': predicted_text})

                    threading.Thread(target=text_to_speech, args=(predicted_text, "hi"), daemon=True).start()

                    last_detection_time = current_time

            except Exception as e:
                print("Prediction Error:", e)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print("Error encoding frame:", e)
            break

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Server starting...")
    socketio.run(app, debug=False, host="127.0.0.1", port=5000, allow_unsafe_werkzeug=True)
