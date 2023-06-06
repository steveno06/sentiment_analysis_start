import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.metrics import accuracy_score
nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

if not nltk.corpus.wordnet.fileids():
    nltk.download('wordnet', quiet=True)


csv_file = 'IMDB_Dataset.csv'
reviews = []

with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        review = row['review']
        sentiment = row['sentiment']
        reviews.append((review, sentiment))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(document):
    tokens = word_tokenize(document.lower())
    preprocessed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            preprocessed_tokens.append(lemmatizer.lemmatize(token))
        
    return preprocessed_tokens

def extract_terms(reviews, category):
    terms = []
    for review, sentiment in reviews:
        if sentiment == category:
            tokens = preprocess(review)
            terms.extend(tokens)
    return terms

#Performing the sentiment analysis

train_size = int(0.8 * len(reviews))
train_reviews = reviews[:train_size]
test_reviews = reviews[train_size:]

positive_terms = extract_terms(train_reviews, 'positive')
negative_terms = extract_terms(train_reviews, 'negative')


positive_term_freq = nltk.FreqDist(positive_terms)
negative_term_freq = nltk.FreqDist(negative_terms)

threshold = 5
lexicon =[]

for term, pos_freq in positive_term_freq.items():
    neg_freq = negative_term_freq.get(term, 0)

    freq_diff = pos_freq - neg_freq

    if freq_diff > threshold:
        lexicon.append((term, freq_diff))
    elif freq_diff < -threshold:
        lexicon.append((term, freq_diff))


lexicon.sort(key=lambda x: x[1], reverse=True)



'''
terms = [term for term, _ in lexicon]
freq_diffs = [freq_diff for _, freq_diff in lexicon]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(freq_diffs, range(len(terms)), color='blue', marker='o')
plt.xlabel('Frequency Difference')
plt.ylabel('')
plt.title('Lexicon Terms and Frequency Differences')

plt.show()
'''
y_true = []
for _, sentiment in test_reviews:
    y_true.append(sentiment)

y_pred = []

for review, _ in test_reviews:
    tokens = preprocess(review)
    sentiment_score = 0
    for term, freq_diff in lexicon:
        if term in tokens:
            sentiment_score = sentiment_score + freq_diff
    
    sentiment = 'positive' if sentiment_score > 0 else 'negative'
    y_pred.append(sentiment)

accuracy = accuracy_score(y_true, y_pred)
print("First 100 words and sentiment scores:")
for i in range(100):
    term, freq_diff = lexicon[i]
    print(f"{term}: {freq_diff}")

# Print the last 100 words and sentiment scores
print("\nLast 100 words and sentiment scores:")
for i in range(len(lexicon) - 100, len(lexicon)):
    term, freq_diff = lexicon[i]
    print(f"{term}: {freq_diff}")
print("Accuracy:", accuracy)
print('Lexicon Length: ', len(lexicon))

