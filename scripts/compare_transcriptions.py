import sys
from pathlib import Path
import transcribe
import transcribe_with_original


def compare_transcriptions(audio_path):
    """Compare transcriptions from fine-tuned and original Whisper models."""
    print("\nTranscribing with fine-tuned model...")
    fine_tuned_result = transcribe.transcribe(audio_path)

    print("\nTranscribing with original model...")
    original_result = transcribe_with_original.transcribe(audio_path)

    print("\nResults Comparison:")
    print("-" * 80)
    print("Fine-tuned model transcription:")
    print(fine_tuned_result)
    print("\nOriginal model transcription:")
    print(original_result)
    print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_transcriptions.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not Path(audio_path).exists():
        print(f"Error: File {audio_path} does not exist")
        sys.exit(1)

    compare_transcriptions(audio_path)
