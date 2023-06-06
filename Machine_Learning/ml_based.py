import csv
import nltk
import numpy as np

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

csv_file= 'IMDB_Dataset.csv'
reviews = []
labels = []

with open(csv_file, "r", encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        review = row['review']
        sentiment = row['sentiment']
        reviews.append(review)
        labels.append(sentiment)

#getting the data ready and preprocessed for the machine learning

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(document):
    tokens = word_tokenize(document.lower())
    preprocessed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            preprocessed_tokens.append(lemmatizer.lemmatize(token))
    # returning a single string of the tokens seperated by a space    
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

num_top_features = 250

top_positive_features = feature_indices[-num_top_features:][::-1]  # get top positive features in descending order
top_negative_features = feature_indices[:num_top_features] # get top negative features in ascending order


feature_names = vectorizer.get_feature_names_out()
sentiment_lexicon = {
    'positive': [feature_names[idx] for idx in top_positive_features],
    'negative': [feature_names[idx] for idx in top_negative_features]
}

def predict_sentiment(review):

    review_tokens = review.split()  

    positive_count = 0
    negative_count = 0
    for token in review_tokens:
        if token in sentiment_lexicon['positive']:
            positive_count += 1
        elif token in sentiment_lexicon['negative']:
            negative_count += 1

    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'  


predictions = []
for review in X_test:
    predicted_sentiment = predict_sentiment(review)
    predictions.append(predicted_sentiment)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)