import pandas as pd
import os

def convert_txt_to_parquet(input_filepath, output_filepath):
    # Takes input path (str) and output path (str), reads tab-separated text and saves as Parquet, returns nothing
    if not os.path.exists(input_filepath):
        print(f"Error: Could not find {input_filepath}")
        return
        
    print(f"Loading {input_filepath}... This might take a moment for large files.")
    
    df = pd.read_csv(input_filepath, sep='\t', on_bad_lines='skip')
    
    print(f"Data loaded successfully. Converting to Parquet...")
    df.to_parquet(output_filepath, engine='pyarrow', compression='snappy')
    
    print(f"Success! Saved compressed Parquet file to {output_filepath}")

if __name__ == "__main__":
    INPUT = './data/raw/amazon_reviews.txt'
    OUTPUT = './data/raw/amazon_reviews.parquet'
    convert_txt_to_parquet(INPUT, OUTPUT)