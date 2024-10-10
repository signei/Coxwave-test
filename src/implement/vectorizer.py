from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def create_vectorizer_and_vectors(questions):
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors

def get_most_similar_questions(query, vectorizer, question_vectors, questions, top_k=5):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(questions[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]

def is_relevant_question(query, vectorizer, question_vectors, threshold=0.1):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    return np.max(similarities) > threshold