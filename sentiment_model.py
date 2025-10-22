# sentiment_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load model once
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    scores = outputs[0][0].detach().numpy()
    scores = softmax(scores)

    labels = ['negative', 'neutral', 'positive']
    result = {labels[i]: float(scores[i]) for i in range(3)}
    sentiment = max(result, key=result.get)
    return sentiment, result
