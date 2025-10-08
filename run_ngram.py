from telugu_ngram_helper import *

def run_model(training=False):
    if not training:
        model = load_model()   #if custom model is there then pass model_file="your_model.pkl"
        if not model:
            training = True
    
    if training:
        tokenized_sentences = load_telugu_data()
        model = NGramModel(n=3)
        model.train(tokenized_sentences)
        model_file = save_ngram_model(model)
        print(f"New model saved as: {model_file}")
        
    
    print(f"Model Info: {len(model.vocab)} words, {model.n}-gram")
    test_phrases = [
    ["భారత", "దేశం"],
    ["తెలుగు", "భాష"], 
    ["హైదరాబాద్", "లో"]
    ]
    
    for phrase in test_phrases:
        preds = model.predict_next(phrase)
        if preds:
            pred_words = [p[0] for p in preds]
            print(f"   '{' '.join(phrase)}' → {pred_words}")
    
    return model

if __name__ == "__main__":
    model = run_model(training=False)