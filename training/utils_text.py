"""
Text processing utilities for summarization training
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    
    return text

def extract_key_sentences(text, num_sentences=3):
    """Extract key sentences based on word frequency"""
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return sentences
    
    # Tokenize into words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Calculate word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score sentences based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence.lower())
        sentence_words = [word for word in sentence_words if word.isalnum()]
        
        score = 0
        for word in sentence_words:
            if word in word_freq:
                score += word_freq[word]
        
        # Normalize by sentence length
        if len(sentence_words) > 0:
            score = score / len(sentence_words)
        
        sentence_scores[i] = score
    
    # Get top sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in sorted_sentences[:num_sentences]]
    top_indices.sort()  # Maintain order
    
    return [sentences[i] for i in top_indices]

def calculate_rouge_simple(reference, hypothesis):
    """
    Simple ROUGE-1 F1 score (word overlap)
    """
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    
    if len(hyp_words) == 0:
        return 0.0
    
    overlap = len(ref_words & hyp_words)
    precision = overlap / len(hyp_words) if len(hyp_words) > 0 else 0
    recall = overlap / len(ref_words) if len(ref_words) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1 * 100

def truncate_text(text, max_words=100):
    """Truncate text to maximum number of words"""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'