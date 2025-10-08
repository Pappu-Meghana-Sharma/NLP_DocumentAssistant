import re
import math
import pickle
import json
import os
from datetime import datetime
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n=3, lambda_ngram=0.8, lambda_unigram=0.2):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.unigram_counts = Counter()
        self.lambda1 = lambda_ngram
        self.lambda3 = lambda_unigram

        total = self.lambda1+self.lambda3
        if abs(total - 1) > 0.001:
            self.lambda1 = 0.7
            self.lambda3 = 0.3

    def train(self, tokenized_sentences):
        self.unigram_counts = Counter()
        self.vocab = set()

        #rewrite this function if json works..remove this part of function
        if tokenized_sentences and isinstance(tokenized_sentences[0], str):
            tokenized_sentences = [self.tokenize_telugu_sentence(s) for s in tokenized_sentences]

        for tokens in tokenized_sentences:
            for word in tokens:
                self.unigram_counts[word] += 1
                self.vocab.add(word)

        for tokens in tokenized_sentences:
            tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                target = tokens[i+self.n-1]
                self.ngrams[context][target] += 1

    def tokenize_telugu_sentence(self, sentence):
        return [w for w in sentence.split() if re.match(r'[\u0C00-\u0C7F]+', w)]

    def predict_next(self, context, top_k=3):
        all_predictions = {}
        unigram_total = sum(self.unigram_counts.values())
        
        candidate_words = set()
        if len(context) >= self.n - 1:
            ngram_context = tuple(context[-(self.n-1):])
            ngram_data = self.ngrams.get(ngram_context, {})
            ngram_total = sum(ngram_data.values())
            
            if ngram_total > 0:
                for word, count in ngram_data.items():
                    if word not in ['<s>', '</s>']:
                        ngram_prob = count / ngram_total
                        all_predictions[word] = self.lambda1 * ngram_prob
                        candidate_words.add(word)

        for word, count in self.unigram_counts.most_common(100):
            if word not in ['<s>', '</s>'] and word not in candidate_words:
                unigram_prob = count / unigram_total
                all_predictions[word] = self.lambda3 * unigram_prob
        
        if all_predictions:
            total_prob = sum(all_predictions.values())
            if total_prob > 0:
                normalized_predictions = {word: prob/total_prob for word, prob in all_predictions.items()}
                sorted_predictions = sorted(normalized_predictions.items(), key=lambda x: x[1], reverse=True)
                return sorted_predictions[:top_k]
        
        return []

    def perplexity(self, test_sentences):
        if test_sentences and isinstance(test_sentences[0], str):
            test_sentences = [self.tokenize_telugu_sentence(s) for s in test_sentences]

        N = 0
        log_prob_sum = 0
        unigram_total = sum(self.unigram_counts.values())

        for tokens in test_sentences:
            tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]
                ngram_prob = 0
                ngram_data = self.ngrams.get(context, {})
                ngram_total = sum(ngram_data.values())
                if ngram_total > 0:
                    ngram_prob = ngram_data.get(word, 0) / ngram_total
                unigram_prob = self.unigram_counts.get(word, 0) / unigram_total
                prob = max(self.lambda1 * ngram_prob + self.lambda3 * unigram_prob, 1e-10)
                log_prob_sum += -math.log(prob)
                N += 1

        return math.exp(log_prob_sum / N) if N > 0 else float('inf')

class TeluguSpellChecker:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def levenshtein_distance(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    def suggest_corrections(self, word, max_suggestions=3):
        if word in self.vocabulary:
            return []
        suggestions = []
        for correct_word in self.vocabulary:
            dist = self.levenshtein_distance(word, correct_word)
            if dist <= 2:
                suggestions.append((correct_word, dist))
        suggestions.sort(key=lambda x: x[1])
        return [word for word, dist in suggestions[:max_suggestions]]

    def check_sentence(self, sentence):
        if isinstance(sentence, str):
            words = [w for w in sentence.split() if re.match(r'[\u0C00-\u0C7F]+', w)]
        else:
            words = sentence

        corrections = {}
        for word in words:
            suggestions = self.suggest_corrections(word)
            if suggestions:
                corrections[word] = suggestions
        return corrections

def tokenize_telugu_sentence(sentence):
    return [w for w in sentence.split() if re.match(r'[\u0C00-\u0C7F]+', w)]

def load_telugu_data():
    sentences = []
    #rewrite this function if json works..remove other part of function accordingly and change the names in case of new files(remove others)
    try:
        with open('telugu_tokenized_sentences.json', 'r', encoding='utf-8') as f:    
            tokenized_sentences = json.load(f)
        print(f"Loaded {len(tokenized_sentences)} pre-tokenized sentences from JSON")
        return tokenized_sentences
    except Exception as e:
        print(f"Pre-tokenized JSON not available and exited with error {e}")

    try:
        with open('Data/final_cleaned_telugu_data.txt', 'r', encoding='utf-8') as f:  #initially this worked...
            sentences = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sentences)} sentences from cleaned data")
    except:
        print("Cleaned data file not available")

    if not sentences:
        try:
            with open('telugu_sentences.txt', 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(sentences)} sentences from sentences file")
        except:
            print("Sentences file not available, using sample data")
            sentences = [
                "రాజు పుస్తకం చదువుతున్నాడు",
                "సూర్యుడు ఉదయమవుతోంది",
                "పిల్లలు పార్క్ లో ఆడుతున్నారు",
                "రాము మరియు నందిని సముద్ర తీరంలో తిరుగుతున్నారు",
                "ఆమె వంటగదిలో కుక్కీ తయారు చేస్తోంది"
            ]

    if sentences and isinstance(sentences[0], str):
        tokenized_sentences = [tokenize_telugu_sentence(s) for s in sentences]
        print(f"Tokenized {len(tokenized_sentences)} sentences")
        return tokenized_sentences

    return sentences

def load_telugu_vocabulary():
    try:
        with open('telugu_vocabulary.txt', 'r', encoding='utf-8') as f:
            vocabulary = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(vocabulary)} words from vocabulary file")
        return vocabulary
    except:
        print("Vocabulary file not available, will use model vocabulary")
        return None

def save_ngram_model(model, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"telugu_ngram_model_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")
    return filename

def load_ngram_model(filename='telugu_ngram_model.pkl'):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    except:
        print(f"Could not load model from {filename}")
        return None

def load_model(model_file="telugu_ngram_model.pkl"):
    if not os.path.exists(model_file):
        return None
    
    return load_ngram_model(model_file)

def complete_sentence(model, context_list, max_len=20):
    completed = context_list.copy()
    for _ in range(max_len):
        if len(completed) < model.n - 1:
            current_context = ["<s>"] * (model.n - 1 - len(completed)) + completed
        else:
            current_context = completed[-(model.n-1):]

        context_tuple = tuple(current_context)
        candidates = model.ngrams.get(context_tuple, {})
        if not candidates:
            if len(completed) > 0:
                context_bi = tuple(completed[-1:])
                candidates = model.ngrams.get(context_bi, {})
        if not candidates:
            break
        total = sum(candidates.values())
        probs = {w: c/total for w, c in candidates.items()}
        next_word = max(probs, key=probs.get)
        if next_word == "</s>":
            break
        completed.append(next_word)
    return completed