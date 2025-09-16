import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = r"C:\slot-A capstone\Dataset-SA.csv"
df = pd.read_csv(file_path)

# Display dataset info
print("Dataset Info:")
print(df.info())

# Drop rows where Review or Sentiment is missing
df = df.dropna(subset=['Review', 'Sentiment'])

# Convert text to lowercase and remove special characters
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['Cleaned_Review'] = df['Review'].astype(str).apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Review'])
y = df['Sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})  # Encode labels

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# Evaluate models
def evaluate_model(y_test, preds, model_name):
    print(f"\n🔹 {model_name} Model Performance:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

evaluate_model(y_test, nb_preds, "Naïve Bayes")
evaluate_model(y_test, svm_preds, "SVM")

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(confusion_matrix(y_test, nb_preds), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Naïve Bayes - Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, svm_preds), annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("SVM - Confusion Matrix")
plt.show()

# Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df["Sentiment"], palette="coolwarm")
plt.title("Sentiment Distribution")
plt.show()

# Generate Word Clouds for Each Sentiment
for sentiment in ['positive', 'negative', 'neutral']:
    text = " ".join(df[df['Sentiment'] == sentiment]['Cleaned_Review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Word Cloud - {sentiment.capitalize()} Sentiment")
    plt.show()

# Sentiment Trend Over Time (if date column is available)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df_time = df.groupby(df['date'].dt.to_period("M"))['Sentiment'].value_counts().unstack()
    df_time.plot(kind='line', figsize=(10,5))
    plt.title("Sentiment Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count of Sentiments")
    plt.show()

# Sentiment Comparison Across Products
df_grouped = df.groupby("product_name")["Sentiment"].value_counts().unstack()
df_grouped.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="coolwarm")
plt.title("Sentiment Comparison Across Products")
plt.xlabel("Product Name")
plt.ylabel("Number of Reviews")
plt.show()

# Sentiment Mapping (if latitude and longitude are available)
if 'latitude' in df.columns and 'longitude' in df.columns:
    df_geo = df.dropna(subset=['latitude', 'longitude'])
    m = folium.Map(location=[df_geo['latitude'].mean(), df_geo['longitude'].mean()], zoom_start=5)

    for _, row in df_geo.iterrows():
        color = "green" if row["Sentiment"] == "positive" else "red" if row["Sentiment"] == "negative" else "blue"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Save and display map
    m.save("sentiment_map.html")
    print("Sentiment Map saved as 'sentiment_map.html'")
