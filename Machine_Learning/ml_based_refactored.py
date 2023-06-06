import csv
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

csv_file = 'IMDB_Dataset.csv'
reviews = []
labels = []

with open(csv_file, "r", encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        review = row['review']
        sentiment = row['sentiment']
        reviews.append(review)
        labels.append(sentiment)

# Getting the data ready and preprocessed for machine learning

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(document):
    tokens = word_tokenize(document.lower())
    preprocessed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            preprocessed_tokens.append(lemmatizer.lemmatize(token))
    # Returning a single string of the tokens separated by a space    
    return ' '.join(preprocessed_tokens) 

preprocessed_reviews = []
for review in reviews:
    preprocessed_review = preprocess(review)
    preprocessed_reviews.append(preprocessed_review)

X_train, X_test, y_train, y_test = train_test_split(preprocessed_reviews, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

feature_indices = np.argsort(classifier.feature_log_prob_[1])

num_top_features = 350

top_positive_features = feature_indices[-num_top_features:][::-1]  # Get top positive features in descending order
top_negative_features = feature_indices[:num_top_features]  # Get top negative features in ascending order

feature_names = vectorizer.get_feature_names_out()
sentiment_lexicon = {}

for idx in top_positive_features:
    feature_name = feature_names[idx]
    positive_log_prob = classifier.feature_log_prob_[1][idx]
    negative_log_prob = classifier.feature_log_prob_[0][idx]
    sentiment_score = positive_log_prob - negative_log_prob

    if sentiment_score > 0:
        sentiment_lexicon[feature_name] = sentiment_score

for idx in top_negative_features:
    feature_name = feature_names[idx]
    positive_log_prob = classifier.feature_log_prob_[1][idx]
    negative_log_prob = classifier.feature_log_prob_[0][idx]
    sentiment_score = positive_log_prob - negative_log_prob

    if sentiment_score < 0:
        sentiment_lexicon[feature_name] = sentiment_score

'''
for word, sentiment_score in sentiment_lexicon.items():
    print(f"{word}: {sentiment_score}")
'''


'''
words = []
scores = []

for word, score in sentiment_lexicon.items():
    words.append(word)
    scores.append(score)

# Sort the words and scores based on the scores
sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
words = [words[i] for i in sorted_indices]
scores = [scores[i] for i in sorted_indices]

# Plot the sentiment scores
plt.figure(figsize=(10, 6))
plt.bar(words, scores)
plt.xlabel('Words')
plt.ylabel('Sentiment Scores')
plt.title('Sentiment Lexicon')
plt.xticks(rotation=90)
plt.show()
'''
def predict_sentiment(review):
    review_tokens = review.split()  

    # Calculate sentiment scores
    positive_score = 0
    negative_score = 0
    for token in review_tokens:
        if token in sentiment_lexicon:
            sentiment_score = sentiment_lexicon[token]
            if sentiment_score > 0:
                positive_score += sentiment_score
            else:
                negative_score += sentiment_score

    #Determine sentiment
    if positive_score > abs(negative_score):
        return 'positive'
    elif negative_score > abs(positive_score):
        return 'negative'
    else:
        return 'neutral'


predictions = []
for review in X_test:
    predicted_sentiment = predict_sentiment(review)
    predictions.append(predicted_sentiment)


accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
