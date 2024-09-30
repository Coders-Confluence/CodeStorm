import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Dataset
df = pd.read_csv('Dataset.csv')

# Feature Creation
df['combined_features'] = df['Details'] + ' ' + df['Benefits'] + ' ' + df['Eligibility']

# Vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['combined_features'])

# Clustering with KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Function to recommend the schemes based on the user's input
def recommend_schemes(user_input):
    user_input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vec, X).flatten()
    indices = similarities.argsort()[-5:][::-1]
    return df.iloc[indices][['Scheme Name', 'Category', 'Details', 'Benefits', 'Eligibility']]

# Example for a user's query
user_query = "I am a mother and I want my daughter to get good education"
recommended_schemes = recommend_schemes(user_query)
print(recommended_schemes)
