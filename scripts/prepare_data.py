import os
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def process_audio(input_path, output_path, target_sr=16000):
    """Process audio file to 16kHz mono WAV format."""
    # Load audio file
    audio, sr = librosa.load(input_path, sr=None, mono=True)

    # Resample if necessary
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Save as WAV
    sf.write(output_path, audio, target_sr)
    return output_path


def prepare_dataset(raw_dir, processed_dir):
    """Prepare dataset by processing audio files and creating required files."""
    # Create directories if they don't exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # Process audio files
    audio_paths = []
    text_entries = []

    # Get all WAV files in raw directory
    wav_files = list(Path(raw_dir).glob("*.wav"))

    print("Processing audio files...")
    for i, wav_path in enumerate(tqdm(wav_files)):
        # Process audio
        output_path = Path(processed_dir) / wav_path.name
        processed_path = process_audio(str(wav_path), str(output_path))

        # Get corresponding text file
        text_path = wav_path.with_suffix('.txt')
        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # Create entries
            unique_id = f"utt_{i:04d}"
            audio_paths.append(f"{unique_id} {processed_path}")
            text_entries.append(f"{unique_id} {text}")

    # Write audio_paths file
    with open('data/audio_paths', 'w', encoding='utf-8') as f:
        f.write('\n'.join(audio_paths))

    # Write text file
    with open('data/text', 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_entries))

    print(f"Processed {len(wav_files)} files")
    print("Created data/audio_paths and data/text files")


if __name__ == "__main__":
    # Create data directories if they don't exist
    for dir_path in ['data/raw', 'data/processed']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    prepare_dataset('data/raw', 'data/processed')
