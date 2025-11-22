import speech_recognition as sr
import os
import sys
import warnings

# Attempt to import libraries for the advanced model
# We wrap this in try-except to make the script robust if the user only wants the basic version
try:
    import torch
    import librosa
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False

class SpeechToTextSystem:
    def __init__(self, use_wav2vec=False):
        """
        Initialize the system.
        :param use_wav2vec: If True, loads the pre-trained Wav2Vec2 model (requires torch).
        """
        self.recognizer = sr.Recognizer()
        self.use_wav2vec = use_wav2vec
        self.wav2vec_model = None
        self.wav2vec_processor = None

        if self.use_wav2vec:
            if not HAS_ADVANCED_LIBS:
                print("[!] Error: Libraries for Wav2Vec (torch, transformers, librosa) not found.")
                print("    Falling back to Google Web Speech API.")
                self.use_wav2vec = False
            else:
                self._load_wav2vec_model()

    def _load_wav2vec_model(self):
        """Loads the Facebook Wav2Vec2 model from Hugging Face."""
        print("[*] Loading pre-trained Wav2Vec2 model... (this may take a moment)")
        try:
            # We use the base-960h model which is a good balance of speed and accuracy
            model_name = "facebook/wav2vec2-base-960h"
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            print("[*] Model loaded successfully.")
        except Exception as e:
            print(f"[!] Failed to load model: {e}")
            self.use_wav2vec = False

    def record_audio(self, duration=5):
        """
        Records audio from the default microphone.
        :param duration: Duration to record in seconds.
        :return: Audio data (SpeechRecognition AudioData object) or None on failure.
        """
        print(f"[*] Recording for {duration} seconds... Speak now!")
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise to handle static
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio_data = self.recognizer.listen(source, timeout=duration + 2, phrase_time_limit=duration)
                print("[*] Recording complete.")
                return audio_data
        except OSError as e:
            print(f"[!] No microphone detected or access denied: {e}")
            return None
        except Exception as e:
            print(f"[!] Error during recording: {e}")
            return None

    def transcribe_audio_data(self, audio_data):
        """
        Transcribes audio data using the selected method.
        :param audio_data: speech_recognition AudioData object
        """
        if self.use_wav2vec:
            return self._transcribe_wav2vec(audio_data)
        else:
            return self._transcribe_google(audio_data)

    def transcribe_file(self, file_path):
        """
        Transcribes an audio file from disk.
        """
        if not os.path.exists(file_path):
            return "[!] File not found."

        print(f"[*] Processing file: {file_path}")
        
        if self.use_wav2vec:
            # Wav2Vec requires loading the raw audio array directly
            return self._transcribe_wav2vec_file(file_path)
        else:
            # SpeechRecognition requires opening the file via its AudioFile class
            try:
                with sr.AudioFile(file_path) as source:
                    audio_data = self.recognizer.record(source)
                return self._transcribe_google(audio_data)
            except Exception as e:
                return f"[!] Error processing file: {e}"

    def _transcribe_google(self, audio_data):
        """Internal method using Google Web Speech API."""
        print("[*] Transcribing via Google Web Speech API...")
        try:
            text = self.recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "[?] Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"[!] Could not request results from Google service; {e}"

    def _transcribe_wav2vec(self, audio_data):
        """
        Internal method to convert AudioData to raw bytes and feed into Wav2Vec.
        Note: This is tricky because AudioData is raw WAV, but Librosa/Torch wants float arrays.
        For simplicity, we save to a temp file and reload, or use the file method directly.
        """
        # Saving to temp file is the safest way to bridge SpeechRecognition and Librosa
        temp_filename = "temp_recording.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_data.get_wav_data())
        
        result = self._transcribe_wav2vec_file(temp_filename)
        
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return result

    def _transcribe_wav2vec_file(self, file_path):
        """Internal method using local Wav2Vec2 model."""
        print("[*] Transcribing via local Wav2Vec2 model...")
        try:
            # 1. Load audio at 16kHz (required by Wav2Vec)
            audio_input, _ = librosa.load(file_path, sr=16000)

            # 2. Tokenize input
            input_values = self.wav2vec_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

            # 3. Perform inference (non-blocking logic for logits)
            with torch.no_grad():
                logits = self.wav2vec_model(input_values).logits

            # 4. Decode predicted ids to text
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]
            
            return transcription.lower() # Wav2Vec usually outputs upper case
            
        except Exception as e:
            return f"[!] Error in Wav2Vec transcription: {e}"

def main():
    print("--- Python Speech-to-Text System ---")
    print("1. Use Google Web Speech API (Requires Internet, Lightweight)")
    print("2. Use Wav2Vec2 Pre-trained Model (Offline, Heavy, High Accuracy)")
    
    choice = input("Select mode (1 or 2): ").strip()
    
    use_advanced = (choice == '2')
    
    system = SpeechToTextSystem(use_wav2vec=use_advanced)
    
    print("\n--- Input Method ---")
    print("A. Record from Microphone")
    print("B. Transcribe Audio File")
    method = input("Select input (A or B): ").strip().upper()
    
    final_text = ""
    
    if method == 'A':
        try:
            seconds = int(input("Enter recording duration in seconds (default 5): ") or 5)
            audio = system.record_audio(duration=seconds)
            if audio:
                final_text = system.transcribe_audio_data(audio)
        except ValueError:
            print("Invalid duration.")
    elif method == 'B':
        path = input("Enter path to .wav file: ").strip()
        # Remove quotes if user dragged and dropped file
        path = path.replace('"', '').replace("'", "")
        final_text = system.transcribe_file(path)
    else:
        print("Invalid selection.")
        return

    print("\n" + "="*40)
    print("TRANSCRIPTION RESULT:")
    print("="*40)
    print(final_text)
    print("="*40)

if __name__ == "__main__":
    main()