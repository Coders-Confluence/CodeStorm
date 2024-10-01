import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

user_details_df = pd.read_csv('Details.csv')

# Dataset
df = pd.read_csv('Dataset.csv')

# Feature Creation
df['combined_features'] = df['Details'] + ' ' + df['Benefits'] + ' ' + df['Eligibility']

# Vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['combined_features'])

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Eligibility
def find_matching_schemes(user_details):
    matching_schemes = []
    
    for _, scheme in df.iterrows():
        # Check if the user qualifies for the scheme based on income, occupation, age, etc.
        if user_details['Income'] <= float(scheme['Eligibility'].split("Income")[1].split()[0]) and \
           user_details['Occupation'] in scheme['Eligibility'] and \
           user_details['Age'] >= int(scheme['Eligibility'].split("ages")[1].split("-")[0]) and \
           user_details['Age'] <= int(scheme['Eligibility'].split("ages")[1].split("-")[1]):
            matching_schemes.append(scheme)

    return pd.DataFrame(matching_schemes)

# Function to recommend schemes based on user input and fetched user details
def recommend_schemes(user_input, user_details):
    # Find schemes based on user details
    eligible_schemes = find_matching_schemes(user_details)
    
    if eligible_schemes.empty:
        return "No matching schemes found based on the given Eligibility Criterias."

    # Vectorize user input
    user_input_vec = vectorizer.transform([user_input])
    
    # Calculate similarity
    X_eligible = vectorizer.transform(eligible_schemes['combined_features'])
    similarities = cosine_similarity(user_input_vec, X_eligible).flatten()
    
    # Get top 5 most similar schemes
    indices = similarities.argsort()[-5:][::-1]
    return eligible_schemes.iloc[indices][['Scheme Name', 'Category', 'Details', 'Benefits', 'Eligibility']]

# Example usage
if __name__ == "__main__":
    # Iterate over each user in the Details CSV
    for _, user_details in user_details_df.iterrows():
        print(f"Recommendations for {user_details['Name']}:")
        
        # Example user query (can be customized per user)
        user_query = "I am looking for financial assistance as a farmer"
        
        # Fetch recommended schemes based on user input and details
        recommended_schemes = recommend_schemes(user_query, user_details)
        print(recommended_schemes)
        print("\n" + "="*80 + "\n")
