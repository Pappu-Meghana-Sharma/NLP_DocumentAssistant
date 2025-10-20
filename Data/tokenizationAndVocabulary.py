import re
import json  

with open("final_cleaned_telugu_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = [s.strip() for s in text.split('.') if s.strip()]

tokenized_sentences = [re.findall(r'[\u0C00-\u0C7F]+', s) for s in sentences]

vocabulary = set(word for sent in tokenized_sentences for word in sent)

with open("telugu_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

with open("telugu_tokenized_sentences.json", "w", encoding="utf-8") as f:
    json.dump(tokenized_sentences, f, ensure_ascii=False, indent=2)

with open("telugu_vocabulary.txt", "w", encoding="utf-8") as f:
    for word in sorted(vocabulary):
        f.write(word + "\n")

print(f"Saved {len(sentences)} sentences to 'telugu_sentences.txt'")
print(f"Saved tokenized sentences to 'telugu_tokenized_sentences.json'")
print(f"Saved {len(vocabulary)} unique words to 'telugu_vocabulary.txt'")
