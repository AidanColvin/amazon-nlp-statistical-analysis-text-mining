import pandas as pd
import os

def load_reviews(path="./data/raw/amazon_reviews.txt"):
    # Takes path (str), handles TSV/CSV formats, returns DataFrame with text and label
    
    try:
        sep = "\t" if path.endswith(".txt") else ","
        df = pd.read_csv(path, sep=sep)
        
        # Ensure standard column names for the rest of the pipeline
        if 'label' in df.columns and 'text' in df.columns:
            return df[['text', 'label']]
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(columns=['text', 'label'])

def verify_file_exists(path):
    # Takes path (str), checks if file is on disk, returns boolean
    
    return os.path.exists(path)