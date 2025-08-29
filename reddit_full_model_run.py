import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk, re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# 1. Load Reddit comments
with open("BI_comments.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split by blank lines (adjust if one comment per line)
comments = [c.strip() for c in text.split("\n\n") if c.strip()]
df = pd.DataFrame({'text': comments})
print("Total comments:", len(df))

# 2. Basic EDA
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
print(df['word_count'].describe())

plt.figure(figsize=(8,5))
# Older Seaborn (<=0.10.x)
sns.distplot(df['word_count'], bins=30, kde=False)  # deprecated but works on old versions

plt.title('Word Count Distribution')
plt.show()

# WordCloud for all comments
wc_all = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
plt.figure(figsize=(10,6))
plt.imshow(wc_all, interpolation='bilinear')
plt.axis('off')
plt.title('Overall Word Cloud')
plt.show()

# 3. Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(txt):
    txt = re.sub(r'http\S+', '', txt)
    txt = re.sub(r'/u/\w+', '', txt)
    txt = re.sub(r'[^a-zA-Z ]', ' ', txt).lower()
    tokens = [lemmatizer.lemmatize(w) for w in txt.split() if w not in stop_words]
    return ' '.join(tokens)

df['clean'] = df['text'].apply(preprocess)

# 4. Sentiment Scoring (VADER)
sia = SentimentIntensityAnalyzer()

def vader_sentiment(txt):
    score = sia.polarity_scores(txt)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['label'] = df['text'].apply(vader_sentiment)
print(df['label'].value_counts())

# WordCloud per sentiment
for sentiment in ['positive', 'neutral', 'negative']:
    wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == sentiment]['clean']))
    plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{sentiment.capitalize()} Word Cloud')
    plt.show()

# 5. Modeling
labeled = df[df['label'] != 'neutral']  # optional: drop neutral
X_train, X_test, y_train, y_test = train_test_split(labeled['clean'], labeled['label'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Baseline models
def eval_model(model):
    model.fit(X_train_tfidf, y_train)
    pred = model.predict(X_test_tfidf)
    print(f"--- {model.__class__.__name__} ---")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Weighted F1:", f1_score(y_test, pred, average='weighted'))
    print("Classification Report:\n", classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print()

for m in [MultinomialNB(), LogisticRegression(max_iter=500), RandomForestClassifier(n_jobs=-1)]:
    eval_model(m)

# 6. Refinement: Grid Search on Logistic Regression
param_grid = {
    'C': [0.1, 1, 5],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train_tfidf, y_train)
best_model = grid.best_estimator_
print("Best Logistic Regression Params:", grid.best_params_)

# Evaluate best model
pred_best = best_model.predict(X_test_tfidf)
print("Best Model Accuracy:", accuracy_score(y_test, pred_best))
print("Best Model F1:", f1_score(y_test, pred_best, average='weighted'))
print(confusion_matrix(y_test, pred_best))

# Save output
df.to_csv("sentiment_results.csv", index=False)
print("Sentiment analysis complete. Results saved to sentiment_results.csv")
