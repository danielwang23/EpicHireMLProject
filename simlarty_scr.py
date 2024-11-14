from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

# Load Sentence Transformer model for semantic embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # This model is efficient for embeddings

# Sample data (usually, these would be extracted keywords or phrases) NEEDS TO BE MODIFIED TO TAKE IN OUTPUT FROM DANIELS RESUME SCRAPER THING
resume_data = {
    "skills": ["Python", "machine learning", "data visualization"],
    "experience": ["data scientist", "analyst", "cloud computing"],
    "education": "Bachelor's in Computer Science"
}

job_data = { 
    "required_skills": ["Python", "data analysis", "visualization"],
    "responsibilities": ["data analysis", "visualization creation"],
    "qualifications": "Bachelor's degree in a relevant field"
}

# Step 1: Function to calculate semantic similarity with multiple methods and weights
def calculate_similarity(list1, list2, cosine_weight=0.7, euclidean_weight=0.3):
    # Generate embeddings for each list
    embeddings1 = sentence_model.encode(list1)
    embeddings2 = sentence_model.encode(list2)

    # Calculate cosine similarity
    cosine_sim_matrix = cosine_similarity(embeddings1, embeddings2)
    max_cosine_sim = np.max(cosine_sim_matrix, axis=1)
    avg_cosine_sim = np.mean(max_cosine_sim)

    # Calculate Euclidean distance (converted to similarity by negating the distance)
    euclidean_dist_matrix = euclidean_distances(embeddings1, embeddings2)
    max_euclidean_sim = np.max(-euclidean_dist_matrix, axis=1)
    avg_euclidean_sim = np.mean(max_euclidean_sim)

    # Combine cosine and Euclidean similarity scores based on weights
    combined_similarity = cosine_weight * avg_cosine_sim + euclidean_weight * avg_euclidean_sim
    
    return combined_similarity

# Step 2: Calculate similarity scores for each category
# Skills similarity score (weighted combination of cosine and Euclidean similarity)
skills_similarity = calculate_similarity(resume_data["skills"], job_data["required_skills"])

# Experience similarity score (weighted combination of cosine and Euclidean similarity)
experience_similarity = calculate_similarity(resume_data["experience"], job_data["responsibilities"])

# Education similarity score
# For education, we'll use a simple binary comparison: 1.0 for exact match, 0.5 for partial match, 0 for no match.
education_similarity = 1.0 if resume_data["education"] == job_data["qualifications"] else 0.5

# Print similarity scores for each category
print("Skills Similarity Score (Combined):", skills_similarity)
print("Experience Similarity Score (Combined):", experience_similarity)
print("Education Similarity Score:", education_similarity)

# Step 3: Store the similarity scores in a dictionary for easy access
similarity_scores = {
    "skills_similarity": skills_similarity,
    "experience_similarity": experience_similarity,
    "education_similarity": education_similarity
}

def calculate_overall_fit(similarity_scores, weights):
    overall_fit_score = (
        weights["skills"] * similarity_scores["skills_similarity"] +
        weights["experience"] * similarity_scores["experience_similarity"] +
        weights["education"] * similarity_scores["education_similarity"]
    )
    return overall_fit_score

# Print the final similarity scores dictionary
print("Similarity Scores:", similarity_scores)
print(overall_fit_score)
