import sys
import torch
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


def load_audio(file_path, sr=16000):
    """Load and preprocess audio file."""
    # Load audio
    audio, sr = librosa.load(file_path, sr=sr)
    return audio


def transcribe(audio_path, model_path="whisper-fine-tuned-final"):
    """Transcribe audio using fine-tuned Whisper model."""
    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Load and preprocess audio
    audio = load_audio(audio_path)

    # Process audio
    input_features = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # Generate transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return transcription


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"Error: File {audio_path} does not exist")
        sys.exit(1)

    result = transcribe(audio_path)
    print("\nTranscription:")
    print(result)
