import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("trained_model.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("model.tflite created successfully!")
