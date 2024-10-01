import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load user details from Details.csv
user_details_df = pd.read_csv('Details.csv')

# Load the dataset of government schemes
df = pd.read_csv('Dataset.csv')

# Combine fields for feature creation
df['combined_features'] = df['Details'] + ' ' + df['Benefits'] + ' ' + df['Eligibility']

# Vectorize the combined text
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['combined_features'])

# Clustering with KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Function to create user query based on their details
def create_user_query(user_details):
    occupation = user_details['Occupation']
    income = user_details['Income']
    age = user_details['Age']
    category = user_details['Category']
    family_members = user_details['Family_Members']
    
    # Create a dynamic query string based on user attributes
    user_query = f"I am a {age} year old {occupation} earning {income} in the {category} category with {family_members} family members."
    
    return user_query

# Function to recommend schemes based on the dynamically generated user query
def recommend_schemes(user_input):
    user_input_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vec, X).flatten()
    indices = similarities.argsort()[-5:][::-1]  # Top 5 schemes
    return df.iloc[indices][['Scheme Name', 'Category', 'Details', 'Benefits', 'Eligibility']]

# Example usage
if __name__ == "__main__":
    # Iterate over each user in the Details CSV
    for _, user_details in user_details_df.iterrows():
        print(f"Recommendations for {user_details['Name']}:")
        
        # Create a dynamic user query from user details
        user_query = create_user_query(user_details)
        
        # Fetch recommended schemes based on the generated query
        recommended_schemes = recommend_schemes(user_query)
        print(recommended_schemes)
        print("\n" + "="*80 + "\n")