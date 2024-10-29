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


def transcribe(audio_path):
    """Transcribe audio using original Whisper tiny model."""
    try:
        # Use the original Whisper tiny model
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny")

        # Load and preprocess audio
        audio = load_audio(audio_path)

        # Process audio
        input_features = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Generate transcription with forced English
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="en", task="transcribe")
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448
        )

        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return transcription
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_with_original.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"Error: File {audio_path} does not exist")
        sys.exit(1)

    result = transcribe(audio_path)
    print(f"TRANSCRIPTION:{result}")
