import re
from syllable import Encoder
from typing import List, Dict, Tuple
from collections import Counter
import math, random

# Load text data in chunks to reduce memory usage
def load_file_in_chunks(filename, chunk_size=1024):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            while True:
                data = file.read(chunk_size)
                if not data:
                    break
                yield data
        print("File read successfully!")
    except Exception as e:
        print("Error reading file:", e)
        return []

# Define the normalization and syllabification functions
def normalize_text(text):
    data = text.lower()
    replacements = {'ç': 'c', 'ş': 's', 'ğ': 'g', 'ö': 'o', 'ü': 'u', 'ı': 'i'}
    for turkish_char, replacement in replacements.items():
        data = data.replace(turkish_char, replacement)
    data = re.sub(r"<.*?>", "", data, flags=re.DOTALL)
    data = re.sub(r'[\u0400-\u04FF\u4E00-\u9FFF\u3400-\u4DBF]', '', data)
    return data

def syllabify_text(text):
    encoder = Encoder()
    cleaned_text = re.sub(r'[\u0400-\u04FF\u4E00-\u9FFF\u3400-\u4DBF]', '', data) # remove russian and chinese words
    cleaned_text = re.sub(r"<.*?>", "", encoder.tokenize(text), flags=re.DOTALL)
    return cleaned_text

# Build character-based and syllable-based N-gram dictionaries
def build_ngram_dict(data: str, n: int) -> Dict[Tuple[str, ...], int]:
    n_gram_dict = Counter(tuple(data[i:i + n]) for i in range(len(data) - n + 1))
    return n_gram_dict

def build_syllable_ngram_dict(data: str, n: int) -> Dict[Tuple[str, ...], int]:
    syllables = data.split()  # Assuming syllabify_text outputs space-separated syllables
    n_gram_dict = Counter(tuple(syllables[i:i + n]) for i in range(len(syllables) - n + 1))
    return n_gram_dict

# Good-Turing smoothing function
def good_turing_smoothing(n_gram_dict: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], float]:
    # Step 1: Count frequencies of frequencies
    freq_of_freqs = Counter(n_gram_dict.values())
    
    # Step 2: Compute adjusted counts for each frequency level
    total_count = sum(n_gram_dict.values())
    smoothed_prob_dict = {}
    
    for n_gram, count in n_gram_dict.items():
        # If there's a count of `count + 1` items, apply smoothing; otherwise use original count
        if (count + 1) in freq_of_freqs and freq_of_freqs[count] > 0:
            adjusted_count = (count + 1) * (freq_of_freqs[count + 1] / freq_of_freqs[count])
            smoothed_prob_dict[n_gram] = adjusted_count / total_count
        else:
            smoothed_prob_dict[n_gram] = count / total_count
    
    # Step 3: Calculate probability for unseen N-grams
    # Probability for unseen items
    unseen_prob = freq_of_freqs[1] / total_count if 1 in freq_of_freqs else 0
    
    return smoothed_prob_dict, unseen_prob

def calculate_perplexity(test_data: str, n: int, smoothed_prob_dict: dict, unseen_prob: float) -> float:
    # Generate N-grams for the test data
    n_grams = [tuple(test_data[i:i + n]) for i in range(len(test_data) - n + 1)]
    
    # Calculate log probability for each N-gram in test data
    log_prob_sum = 0
    total_ngrams = len(n_grams)
    
    for n_gram in n_grams:
        # Use smoothed probability if available, else use the unseen probability
        prob = smoothed_prob_dict.get(n_gram, unseen_prob)
        log_prob_sum += math.log(prob)
    
    # Perplexity formula
    perplexity = math.exp(-log_prob_sum / total_ngrams)
    return perplexity

# Function to get top 5 N-grams by probability
def get_top_5_ngrams(ngram_dict, prefix):
    candidates = [(k, v) for k, v in ngram_dict.items() if k[:-1] == prefix]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_5 = candidates[:5]
    total_count = sum(count for _, count in top_5)
    top_5_probs = [(k[-1], count / total_count) for k, count in top_5]
    return top_5_probs

# Generate sentences using N-grams
def generate_sentence(ngram_dict, n, length=50, syllable_based=False):
    if n == 1:
        # For 1-gram model, pick characters/syllables independently
        elements = list(ngram_dict.keys())
        probabilities = [ngram_dict[element] / sum(ngram_dict.values()) for element in elements]
        sentence = random.choices(elements, probabilities, k=length)
    else:
        # Start with a random prefix for 2-gram or 3-gram model
        sentence = list(random.choice(list(ngram_dict.keys()))[:n-1])
        
        for _ in range(length - (n - 1)):
            prefix = tuple(sentence[-(n-1):])  # Get last (n-1) elements as prefix
            top_5_candidates = get_top_5_ngrams(ngram_dict, prefix)
            
            if not top_5_candidates:
                break  # Stop if no candidates are available

            # Select next element from top 5 candidates randomly, weighted by probability
            next_element = random.choices([c[0] for c in top_5_candidates],
                                          [c[1] for c in top_5_candidates])[0]
            # Convert next_element to a string if it's a tuple, and append it to sentence
            if isinstance(next_element, tuple):
                sentence.extend(next_element)  # Add each element in the tuple as a separate string
            else:
                sentence.append(next_element)

    # Join characters or syllables
    return sentence



# Load data and apply transformations
data = ""
for chunk in load_file_in_chunks("wiki_00", chunk_size=10240 * 1024):  # 1 MB chunks
    data += chunk

# Apply normalization and syllabification
normalized_data = normalize_text(data)
syllabified_data = syllabify_text(data)

# Use only a subset for training/testing to reduce memory load
train_normalized_data = normalized_data[:int(0.95 * len(normalized_data))]
train_syllabified_data = syllabified_data[:int(0.95 * len(syllabified_data))]

test_normalized_data = normalized_data[int(0.95 * len(normalized_data)):]
test_syllabified_data = syllabified_data[int(0.95 * len(syllabified_data)):]

## Calculate perplexity for 1-gram, 2-gram, and 3-gram models in both character-based and syllable-based models
for n in [1, 2, 3]:
   # Character-based model perplexity
    character_ngram_dict = build_ngram_dict(train_normalized_data, n)
    character_ngram_smoothed, character_unseen_prob = good_turing_smoothing(character_ngram_dict)
    character_perplexity = calculate_perplexity(test_normalized_data, n, character_ngram_smoothed, character_unseen_prob)
    
    print(f"Character-Based {n}-Gram preplexity:")
    print(character_perplexity)
    print(f"Character-Based {n}-Gram sentence:")
    print(generate_sentence(character_ngram_smoothed, n))
    
    
    # Syllable-based model perplexity
    syllable_ngram_dict = build_syllable_ngram_dict(train_syllabified_data, n)
    syllable_ngram_smoothed, syllable_unseen_prob = good_turing_smoothing(syllable_ngram_dict)
    syllable_perplexity = calculate_perplexity(test_syllabified_data, n, syllable_ngram_smoothed, syllable_unseen_prob)
    
    print(f"Syllable-Based {n}-Gram preplexity:")
    print(syllable_perplexity)
    print(f"Syllable-Based {n}-Gram sentence:")
    print(generate_sentence(syllable_ngram_smoothed, n, syllable_based=True))
    
