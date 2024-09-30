from supabase import create_client, Client
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Supabase connection details
SUPABASE_URL = "https://pwhuldcdbdjtutkapfdu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB3aHVsZGNkYmRqdHV0a2FwZmR1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mjc3MTg4MTAsImV4cCI6MjA0MzI5NDgxMH0.udgaH1AxkAWfXKkFuqaL2kIfSO1VxoFSWUFt-5UnzOc"

# Create a Supabase client
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch user details from Supabase
def fetch_user_details(supabase: Client, user_id: str):
    response = supabase.table('user_details').select('*').eq('user_id', user_id).execute()
    return response.data[0]  # Returns a dictionary of user details

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

# Cross-reference user details with the schemes dataset for eligibility
def find_matching_schemes(user_details):
    matching_schemes = []
    
    for _, scheme in df.iterrows():
        # Check if the user qualifies for the scheme based on income, occupation, age, etc.
        if user_details['income'] <= float(scheme['Income Limit']) and \
           user_details['occupation'] in scheme['Eligible Occupation'] and \
           user_details['age'] >= int(scheme['Min Age']) and \
           user_details['age'] <= int(scheme['Max Age']):
            matching_schemes.append(scheme)

    return pd.DataFrame(matching_schemes)

# Function to recommend schemes based on user input and fetched user details
def recommend_schemes(user_input, user_details):
    # Find schemes based on user details
    eligible_schemes = find_matching_schemes(user_details)
    
    if eligible_schemes.empty:
        return "No matching schemes found based on eligibility."

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
    # Initialize Supabase client
    supabase = get_supabase_client()

    # Example user details fetch (replace 'user-unique-id' with actual user ID)
    user_details = fetch_user_details(supabase, "user-unique-id")
    
    # Example user query
    user_query = "I am a mother and I want my daughter to get good education"
    
    # Fetch recommended schemes based on user input and details
    recommended_schemes = recommend_schemes(user_query, user_details)
    print(recommended_schemes)
