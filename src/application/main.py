import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.implement.data_loader import load_faq_data
from src.implement.vectorizer import create_vectorizer_and_vectors
from src.implement.chatbot import run_chatbot

CONFIG = {
    'openai_api_key': "api-key",
    'faq_data_path': os.path.join(project_root, 'src', 'resource', 'final_result.pkl'),
    'model_name': 'gpt-3.5-turbo',
    'max_tokens': 150,
}

def main():
    faq_data = load_faq_data(CONFIG['faq_data_path'])
    questions = list(faq_data.keys())
    vectorizer, question_vectors = create_vectorizer_and_vectors(questions)
    
    run_chatbot(CONFIG, faq_data, vectorizer, question_vectors)

if __name__ == "__main__":
    main()