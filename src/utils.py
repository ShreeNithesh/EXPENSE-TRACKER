import pandas as pd
import numpy as np
import re

def clean_text(text):
    """Clean merchant description text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def sample_data(df, sample_size=100000):
    """Sample data for training to handle large dataset."""
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42)
    return df