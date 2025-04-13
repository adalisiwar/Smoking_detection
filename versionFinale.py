import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

hand_detector = HandDetector(detectionCon=0.8, maxHands=2)

# l video li bch ntestiw bih
cap = cv2.VideoCapture(r"C:\Users\adali\OneDrive\Desktop\computer vision project\smoker_guy.mp4")

# cigarette color detection 
lower_orange = np.array([10, 160, 160])
upper_orange = np.array([20, 255, 255])
lower_red = np.array([170, 120, 120])
upper_red = np.array([180, 255, 255])

# Smoke detection color range
lower_smoke = np.array([0, 0, 200])  
upper_smoke = np.array([180, 50, 255])

# Morphological kernel for noise removal 
kernel = np.ones((3, 3), np.uint8)

# taba3 cigarette detection over multiple frames
cigarette_frame_count = 0
cigarette_threshold = 5

while True:
    ret, frame = cap.read()

    if not ret:  
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        continue


    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Hand detection
    hands, _ = hand_detector.findHands(frame, draw=False)

    if hands:
        for hand in hands:
            # Draw bounding box around the detected hand
            x, y, w, h = hand['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, "Hand", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Convert frame to HSV for color-based cigarette detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    cigarette_mask = cv2.bitwise_or(mask_orange, mask_red)

    # Remove noise using morphological operations
    cigarette_mask = cv2.morphologyEx(cigarette_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the cigarette-like objects
    contours, _ = cv2.findContours(cigarette_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cigarette_detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Increase minimum area threshold
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Check if the object has an aspect ratio that suggests it's a cigarette
            if 2.5 < aspect_ratio < 5:  # Narrow aspect ratio range
                cigarette_detected = True
                # Draw a green bounding box around the detected cigarette
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
                cv2.putText(frame, "Cigarette", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break

    # Convert frame to HSV for color-based smoke detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for smoke detection
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

    # Remove noise using morphological operations
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the smoke-like objects (logic retained, but no bounding boxes or labels)
    smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in smoke_contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Minimum area threshold for smoke
            # Logic for smoke detection is retained, but no bounding boxes or labels are drawn
            pass

    # If hands are detected, check for pinch between thumb and index finger bch n3rfou ken cigarette is detected 
    if hands:
        hand = hands[0]
        thumb_tip = hand['lmList'][4]
        index_tip = hand['lmList'][8]

        # Calculate the distance between thumb and index finger tips
        distance = np.linalg.norm(np.array(thumb_tip[:2]) - np.array(index_tip[:2]))

        # If thumb and index are close and cigarette is detected, mark as smoking
        if distance < 40 and cigarette_detected:
            cigarette_frame_count += 1
        else:
            cigarette_frame_count = 0

        cv2.putText(frame, "Smoking detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # taffichi resultat in real-time
    cv2.imshow("Real-Time Smoking Detection", frame)

    # tnzel echap bch to5rj
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
