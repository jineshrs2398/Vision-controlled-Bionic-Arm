# Vision-Controlled Bionic Arm

## Project Overview
This repository contains the code and resources for the Vision-Controlled Bionic Arm project, an innovative system that uses real-time hand gesture recognition to control a robotic arm. The project leverages Python for gesture recognition through MediaPipe and controls servo motors via Arduino to mimic these gestures physically.

## Features
- Real-time hand gesture tracking using MediaPipe
- Servo control through Arduino for precise arm movements
- Integration of LSM9DS1 sensor to refine movements based on gyroscopic and accelerometer data
- Robust error handling and system recovery mechanisms

## Getting Started
### Prerequisites
- Python 3.8+
- Arduino IDE
- MediaPipe
- OpenCV for Python
- Adafruit LSM9DS1 library
- Adafruit Servo Driver library

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/vision-controlled-bionic-arm.git
```

2. **Install the required Python libraries:**
```bash
pip install opencv-python mediapipe numpy
```

3. **Set up the Arduino environment:**
- Install the Adafruit Servo Driver and LSM9DS1 libraries through the Arduino Library Manager.

### Hardware Setup
- Connect the LSM9DS1 sensor and servo drivers to your Arduino according to the circuit diagrams provided in the `hardware` directory.

### Running the Project
1. **Launch the Python script to begin gesture tracking:**
```bash
python app.py
```
2. **Upload the Arduino sketch to the microcontroller.**
3. **Perform hand gestures in front of the camera to control the bionic arm.**


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

