# AI Emotion Detector using DeepFace

A real-time facial emotion recognition system built with Python, OpenCV, and DeepFace.

The application captures webcam video, detects facial emotions using a deep learning model, and displays personalized motivational messages based on the detected emotion.

## Features

* Real-time webcam emotion detection
* DeepFace-powered facial analysis
* Emotion-specific motivational messages
* Live countdown timer
* Optimized inference interval for smoother performance
* User-friendly visual interface

## Technologies Used

* Python
* OpenCV
* DeepFace
* TensorFlow

## Supported Emotions

* Happy 😊
* Sad 😔
* Angry 😠
* Surprise 😲
* Fear 😨
* Neutral 😐

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Emotion-Detector.git
cd AI-Emotion-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python emotion_detector.py
```

## How It Works

1. Captures live video using OpenCV.
2. Performs facial emotion analysis using DeepFace.
3. Detects the dominant emotion.
4. Displays emotion-specific messages.
5. Automatically stops after 15 seconds or when the user presses Q.

## Performance Optimization

To improve responsiveness, emotion analysis is performed at fixed intervals rather than on every frame. This significantly reduces CPU/GPU load while maintaining a smooth user experience.

## Future Enhancements

* Emotion-based music recommendation
* Emotion analytics dashboard
* Emotion history tracking
* Multi-face emotion detection
* Emotion-based chatbot integration

## Author

Harsh Joshi

## License

MIT License
