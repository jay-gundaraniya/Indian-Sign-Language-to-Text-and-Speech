import cv2
import os

# Dataset directory
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Ask the user for the word (Hindi word in English letters)
word = input("Enter the Hindi word (in English letters) for this sign: ").strip()
word_dir = os.path.join(DATASET_DIR, word)
os.makedirs(word_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam!")
    exit()

print(f"📸 Capturing images for '{word}'... Press 's' to save, 'q' to quit.")

image_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame!")
        break

    # Show the frame
    cv2.putText(frame, f"Sign: {word} ({image_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Capture Sign Language Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save image
        image_path = os.path.join(word_dir, f"{word}_{image_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved: {image_path}")
        image_count += 1
    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
print(f"{image_count} images saved for '{word}'.")
  