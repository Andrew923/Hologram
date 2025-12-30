import os
# Set pure-Python protobuf implementation for compatibility with locally built mediapipe
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp

# --- Configuration ---
INPUT_FILE = "capture.jpg"   # The file you just took with the camera
OUTPUT_FILE = "result.jpg"   # The output file with drawings

# Check if input exists
if not os.path.exists(INPUT_FILE):
    print(f"Error: Could not find '{INPUT_FILE}'. Did you run the camera snap script?")
    exit()

print(f"Loading {INPUT_FILE}...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Run MediaPipe
# static_image_mode=True is important for processing single files (it's more accurate but slower)
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    # Read the image
    image = cv2.imread(INPUT_FILE)
    
    # Convert BGR (OpenCV format) to RGB (MediaPipe format)
    # MediaPipe will CRASH or give bad results if you forget this step!
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    print("Running MediaPipe inference...")
    results = hands.process(image_rgb)

    # Check results
    if not results.multi_hand_landmarks:
        print("No hands were detected in the image.")
    else:
        print(f"Success! Detected {len(results.multi_hand_landmarks)} hand(s).")
        
        # Draw the landmarks on the image
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Save the result
        cv2.imwrite(OUTPUT_FILE, annotated_image)
        print(f"Saved annotated image to: {OUTPUT_FILE}")