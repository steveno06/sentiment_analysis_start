import csv
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

reviews = []
sentiments = []

with open('IMDB_Dataset.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        reviews.append(row['review'])
        sentiments.append(row['sentiment'])

train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(reviews, sentiments, test_size=0.2, random_state=42)

stop_words = stopwords.words('english')
analyzer = SentimentIntensityAnalyzer()

threshold = 0.2
movie_specific_words = {}

for review, sentiment in zip(train_reviews, train_sentiments):
    words = word_tokenize(review.lower())
    preprocessed_tokens = []
    for token in words:
        if token.isalpha() and token not in stop_words:
            preprocessed_tokens.append(token)
    words = preprocessed_tokens

    for word in words:
        if word not in movie_specific_words:
            movie_specific_words[word] = {'positive':0, 'negative':0}
        scores = analyzer.polarity_scores(word)
        if sentiment == 'positive':
            movie_specific_words[word]['positive'] += scores['pos']
        elif sentiment == 'negative':
            movie_specific_words[word]['negative'] += scores['neg']

lexicon_combined = analyzer.lexicon.copy()
lexicon_combined.update(movie_specific_words)

correct_predictions_combined = 0

for review, sentiment in zip(test_reviews, test_sentiments):
    words = word_tokenize(review.lower())
    preprocessed_tokens = []
    for token in words:
        if token.isalpha() and token not in stop_words:
            preprocessed_tokens.append(token)
    words = preprocessed_tokens

    pos_score = {}
    neg_score = {}

    for word in words:
        if word in lexicon_combined:
            if isinstance(lexicon_combined[word], dict):
                if 'positive' in lexicon_combined[word]:
                    pos_score[word] = pos_score.get(word, 0) + lexicon_combined[word]['positive']
                if 'negative' in lexicon_combined[word]:
                    neg_score[word] = neg_score.get(word, 0) + lexicon_combined[word]['negative']

    total_pos_score = sum(pos_score.values())
    total_neg_score = sum(neg_score.values())
    
    prediction = 'positive' if total_pos_score > total_neg_score else 'negative'
    if prediction == sentiment:
        correct_predictions_combined += 1

accuracy_combined = correct_predictions_combined / len(test_reviews)
print("Accuracy using the combined lexicon on the testing set: {:.2%}".format(accuracy_combined))
