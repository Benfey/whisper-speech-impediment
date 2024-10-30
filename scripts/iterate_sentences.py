import json


class HarvardSentences:
    def __init__(self, json_file='harvard_sentences.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.sentences = self.data['sentences']
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.sentences):
            raise StopIteration
        sentence = self.sentences[self.current_index]
        self.current_index += 1
        return sentence

    def reset(self):
        self.current_index = 0

    def get_sentence(self, index):
        if 0 <= index < len(self.sentences):
            return self.sentences[index]
        raise IndexError("Index out of range")

    def total_sentences(self):
        return len(self.sentences)


# Example usage
if __name__ == "__main__":
    # Create an instance
    harvard = HarvardSentences()

    print(f"Total sentences: {harvard.total_sentences()}")

    # Example 1: Using iterator
    print("\nExample 1: First 5 sentences using iterator:")
    for i, sentence in enumerate(harvard):
        if i >= 5:
            break
        print(f"{i+1}. {sentence}")

    # Example 2: Direct access
    print("\nExample 2: Accessing specific sentence:")
    print(f"Sentence #10: {harvard.get_sentence(9)}")  # 0-based index

    # Example 3: Reset and iterate again
    harvard.reset()
    print("\nExample 3: Iterator reset and first 3 sentences:")
    for i, sentence in enumerate(harvard):
        if i >= 3:
            break
        print(f"{i+1}. {sentence}")
