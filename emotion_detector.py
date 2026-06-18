import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import random
from collections import Counter
from deepface import DeepFace

# ----------------------------
# Emotion Messages
# ----------------------------
emotion_messages = {
    "happy": [
        "Your positive energy is inspiring.",
        "Keep spreading those good vibes.",
        "A smile can brighten any room."
    ],
    "sad": [
        "Every day is a fresh beginning.",
        "Tough moments create strong people.",
        "Keep moving forward."
    ],
    "angry": [
        "Take a deep breath and reset.",
        "Patience is a hidden strength.",
        "Stay calm and stay focused."
    ],
    "surprise": [
        "Looks like something caught your attention.",
        "Unexpected moments make life exciting.",
        "Curiosity leads to discovery."
    ],
    "fear": [
        "You are stronger than you think.",
        "Courage grows one step at a time.",
        "Keep believing in yourself."
    ],
    "neutral": [
        "Focused and ready for action.",
        "A calm mind makes better decisions.",
        "Steady progress wins the race."
    ],
    "disgust": [
        "Stay composed and keep moving.",
        "Every reaction teaches something.",
        "Focus on what matters most."
    ]
}

# ----------------------------
# Camera Setup
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to access webcam.")
    exit()

# ----------------------------
# Session Variables
# ----------------------------
session_duration = 20  # seconds
analysis_interval = 2.5

start_time = time.time()
last_analysis_time = 0

emotion = "detecting"
confidence = 0.0
message = "Initializing AI Emotion Detector..."

emotion_counter = Counter()

# ----------------------------
# Main Loop
# ----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera Error.")
        break

    frame = cv2.flip(frame, 1)

    current_time = time.time()
    elapsed_time = int(current_time - start_time)
    remaining_time = max(0, session_duration - elapsed_time)

    # Run DeepFace periodically
    if current_time - last_analysis_time > analysis_interval:
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotion = result["dominant_emotion"]
            confidence = result["emotion"][emotion]

            emotion_counter[emotion] += 1

            message = random.choice(
                emotion_messages.get(
                    emotion,
                    ["Stay positive and keep learning."]
                )
            )

        except Exception:
            pass

        last_analysis_time = current_time

    # Most frequent emotion
    most_seen = "N/A"
    if emotion_counter:
        most_seen = emotion_counter.most_common(1)[0][0]

    # Background panel
    cv2.rectangle(frame, (10, 10), (620, 220), (40, 40, 40), -1)

    # Emotion
    cv2.putText(
        frame,
        f"Emotion: {emotion.upper()}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence:.1f}%",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Most frequent emotion
    cv2.putText(
        frame,
        f"Most Seen: {most_seen.upper()}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # Timer
    cv2.putText(
        frame,
        f"Session Time Left: {remaining_time}s",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2
    )

    # Message
    cv2.putText(
        frame,
        message,
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 0),
        2
    )

    cv2.imshow("AI Emotion Detector Pro", frame)

    # Auto-stop
    if elapsed_time >= session_duration:
        print("\nSession Complete!")
        break

    # Manual quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------
# Save Session Report
# ----------------------------
with open("emotion_log.txt", "w", encoding="utf-8") as f:
    f.write("AI Emotion Detector - Session Report\n")
    f.write("=" * 40 + "\n\n")

    total_detections = sum(emotion_counter.values())

    f.write(f"Total Emotion Analyses: {total_detections}\n\n")

    for emotion_name, count in emotion_counter.items():
        percentage = (count / total_detections) * 100 if total_detections else 0
        f.write(
            f"{emotion_name.capitalize():<12}: "
            f"{count} detections ({percentage:.1f}%)\n"
        )

    if emotion_counter:
        dominant = emotion_counter.most_common(1)[0][0]
        f.write(f"\nDominant Emotion: {dominant.upper()}\n")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Emotion report saved as 'emotion_log.txt'")
