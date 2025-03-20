from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import threading
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import time

warnings.filterwarnings("ignore",
                        message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    label_map = model_dict['label_map']
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)
    model = None
    label_map = {}

last_spoken_text = None
detection_cooldown = 1.5
last_detection_time = 0
last_prediction_time = 0 
prediction_interval = 0.5 

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


def text_to_speech(text, lang='hi'):
    global last_spoken_text, last_detection_time

    current_time = time.time()
    if text.strip() and (text != last_spoken_text or (current_time - last_detection_time) > detection_cooldown):
        last_spoken_text = text
        last_detection_time = current_time
        print("Speaking:", text)

        def play_audio():
            tts = gTTS(text=text, lang=lang)
            tts.save("temp_audio.mp3")
            data, samplerate = sf.read("temp_audio.mp3")
            sd.play(data, samplerate)
            sd.wait()

        threading.Thread(target=play_audio).start()


def generate_frames():
    global last_prediction_time
    print("Starting Video Feed...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Cannot access webcam!")
        return

    print("Camera Accessed Successfully!")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame!")
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        current_time = time.time()
        if results.multi_hand_landmarks and (current_time - last_prediction_time) > prediction_interval:
            last_prediction_time = current_time

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

            def predict_and_speak():
                try:
                    prediction = model.predict([np.asarray(data_aux)])[0]
                    predicted_text = label_map.get(prediction, "Unknown")
                    if predicted_text != last_spoken_text:
                        print("New Sign Detected:", predicted_text)
                    socketio.emit('prediction', {'text': predicted_text})
                    text_to_speech(predicted_text)
                except Exception as e:
                    print("Prediction Error:", e)

            threading.Thread(target=predict_and_speak).start()

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print("Error encoding frame:", e)
            break


@app.route('/video_feed')
def video_feed():
    print("Serving video feed...")
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("ERROR in video feed:", e)
        return "Error loading video feed", 500


if __name__ == '__main__':
    socketio.run(app, debug=True)