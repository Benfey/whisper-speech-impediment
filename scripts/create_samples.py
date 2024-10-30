import json
import os

# Read the JSON file
with open('data/raw/harvard_sentences.json', 'r') as f:
    data = json.load(f)

# Create output directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Write each sentence to its own file
for i, sentence in enumerate(data['sentences'], 1):
    filename = f'data/raw/sample{i:03d}.txt'
    with open(filename, 'w') as f:
        f.write(sentence)

print(f"Created {len(data['sentences'])} sample files.")
