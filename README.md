# Hybrid Speech-to-Text Transcriber

A robust, dual-engine speech-to-text system built in Python. This project solves the trade-off between speed and privacy by allowing users to toggle between a lightweight Cloud API and a high-accuracy Offline Deep Learning Model. :contentReference[oaicite:1]{index=1}

## Features

### **Dual Engine Architecture**
- **Cloud Mode:** Uses Google Web Speech API for fast, lightweight results (requires internet).
- **Offline AI Mode:** Uses Facebook's Wav2Vec2 model (via Hugging Face) for secure, high-accuracy transcription without sending data to the cloud.

### **Flexible Inputs**
- **Real-time Microphone:** Record and transcribe live audio directly from the terminal.
- **File Processing:** Upload and transcribe existing `.wav` files.

### **Smart Preprocessing**
- Automatically resamples input audio to 16kHz to match model requirements.

### **Automatic Fallback**
- Gracefully defaults to the cloud API if deep learning libraries are missing.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**
- **A working microphone** (for live recording)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/speech-to-text-system.git
cd speech-to-text-system
```
2. Install dependencies:
```bash
pip install SpeechRecognition pyaudio transformers torch librosa soundfile
```

*Note for Linux/Mac Users:* If you encounter errors installing `pyaudio`, you may need to install the system-level PortAudio libraries first: 
  + Ubuntu/Debian: `sudo apt-get install python3-pyaudio portaudio19-dev`
  + macOS: `brew install portaudio`

## Usage
Run the main script from your terminal:
```base
python speech_transcriber.py
```
follow the interactive prompts to choose your mode and inpur source.

**Example Workflow**
```bash
--- Python Speech-to-Text System ---
1. Use Google Web Speech API (Requires Internet, Lightweight)
2. Use Wav2Vec2 Pre-trained Model (Offline, Heavy, High Accuracy)
Select mode (1 or 2): 2

 Loading pre-trained Wav2Vec2 model...
 Model loaded successfully.

--- Input Method ---
A. Record from Microphone
B. Transcribe Audio File
Select input (A or B): A

 Recording for 5 seconds... Speak now!
 Transcription: "Hello world this is a test"
```
## How It Works

1. Input Handling: The system captures audio via `SpeechRecognition` (mic) or `librosa` (file).

2. **Engine Selection:**

+ If Google API is selected, the audio buffer is sent to Google's STT servers.
+ If Wav2Vec2 is selected, the audio is converted to a floating-point array, resampled to     16,000Hz, and passed through a local PyTorch neural network.

3. **Decoding:** The model's logits (raw predictions) are decoded into readable text and displayed.

## Contributing 
Contributions are welcome! Please follow these steps:

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add some NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

## License
Distributed under the MIT License. See `LICENSE` for more information.
