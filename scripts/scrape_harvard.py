import requests
from bs4 import BeautifulSoup
import json
import re


def scrape_harvard_sentences():
    # Get the webpage content
    url = "https://www.cs.columbia.edu/~hgs/audio/harvard.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all sentences (they are in li elements)
    sentences = []
    list_items = soup.find_all('li')

    for li in list_items:
        text = li.get_text().strip()
        if text:  # If there's any text
            # Remove any leading numbers if present
            cleaned_text = re.sub(r'^\d+\.\s*', '', text)
            sentences.append(cleaned_text)

    # Save to JSON file
    with open('harvard_sentences.json', 'w', encoding='utf-8') as f:
        json.dump({'sentences': sentences}, f, indent=4)

    print(f"Saved {len(sentences)} sentences to harvard_sentences.json")


if __name__ == "__main__":
    scrape_harvard_sentences()
