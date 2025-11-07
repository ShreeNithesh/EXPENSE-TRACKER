import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder
from src.utils import clean_text, sample_data

# Optional Keras tokenizer (only used if TensorFlow/Keras is installed)
try:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    Tokenizer = None
    pad_sequences = None

class DataPreprocessor:
    def load_data(self, file_path):
        """Load large CSV in chunks to handle size."""
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded dataset with {len(df)} rows.")
        return df

    def preprocess_data(self, df):
        """Preprocess data: clean text, handle missing, feature engineering."""
        df = df.dropna(subset=['merchant', 'category', 'amt'])
        df['merchant_clean'] = df['merchant'].apply(clean_text)
        df['amt_log'] = np.log1p(df['amt'])
        df = sample_data(df)
        label_encoder = SKLabelEncoder()
        label_encoder.fit(df['category'])
        df['category_encoded'] = label_encoder.transform(df['category'])
        print(f"Preprocessed dataset with {len(df)} rows.")
        return df, label_encoder

    def create_features(self, df, max_features=1000, max_len=20):
        """Create enhanced TF-IDF features from merchant and scale numerical.

        Returns:
            X, y, vectorizer, scaler, tokenizer, sequences
        """
        # Enhanced TF-IDF with better parameters for precise categorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            stop_words='english',  # Remove common English stop words
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only words with 2+ letters
        )
        tfidf_matrix = vectorizer.fit_transform(df['merchant_clean']).toarray()

        scaler = SKStandardScaler()
        # Add more numerical features for better prediction
        numerical_features = df[['amt_log']].values
        numerical_scaled = scaler.fit_transform(numerical_features)

        X = np.hstack([tfidf_matrix, numerical_scaled])
        y = df['category_encoded'].values

        tokenizer = None
        sequences = None
        if Tokenizer is not None:
            tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
            tokenizer.fit_on_texts(df['merchant_clean'])
            seqs = tokenizer.texts_to_sequences(df['merchant_clean'])
            sequences = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')

        print(f"Enhanced feature matrix shape: {X.shape}")
        print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
        return X, y, vectorizer, scaler, tokenizer, sequences