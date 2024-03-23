import spacy
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text data
def preprocess_text(text):
    # Process the review text
    doc = nlp(text)
    # Remove stopwords and punctuation
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    # Join tokens back into a string
    clean_text = ' '.join(tokens)
    return clean_text

# Function for sentiment analysis
def analyze_sentiment(review):
    # Preprocess the review text
    clean_review = preprocess_text(review)
    # Analyze sentiment using spaCy
    doc = nlp(clean_review)
    polarity = doc.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Load dataset
data = pd.read_csv('amazon_product_reviews.csv')

# Drop rows with missing review text
clean_data = data.dropna(subset=['review.text'])

# Test sentiment analysis function on sample reviews
for i, review in enumerate(clean_data['review.text'], 1):
    sentiment = analyze_sentiment(review)
    print(f"Review {i}: {review} - Sentiment: {sentiment}")
