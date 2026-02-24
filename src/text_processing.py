import re
import os
import string

def remove_punctuation(text):
    # Takes text (str), removes punctuation, returns clean text (str)
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_whitespace_and_lowercase(text):
    # Takes text (str), removes extra whitespace and converts to lowercase, returns clean text (str)
    return re.sub(r'\s+', ' ', text).lower().strip()

def get_stop_words():
    # Takes nothing, defines English stop words, returns set of stop words (set)
    return {
        'a','an','the','and','or','but','if','while','with','of','at','by','for',
        'to','from','in','on','out','over','under','again','further','then','once',
        'here','there','when','where','why','how','all','any','both','each','few',
        'more','most','other','some','such','no','nor','not','only','own','same',
        'so','than','too','very','can','will','just','is','are','was','were','be',
        'been','being','have','has','had','do','does','did','this','that','these',
        'those','i','me','my','we','our','you','your','he','him','his','she','her',
        'it','its','they','them','their'
    }

def process_text(text):
    # Takes text (str), removes punctuation/case/stop words, returns list of words (list)
    text = remove_punctuation(text)
    text = clean_whitespace_and_lowercase(text)
    stop_words = get_stop_words()
    return [w for w in text.split() if w not in stop_words]

def extract_label_and_text(line):
    # Takes file line (str), splits by tab, returns label (str) and text (str)
    parts = line.strip().split('\t')
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]

def load_dictionary(filepath):
    # Takes filepath (str), reads valid ascii words, returns list of words (list)
    words = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if ' ' not in word and word.isascii() and word.isalpha() and len(word) <= 20:
                    words.append(word)
    return words