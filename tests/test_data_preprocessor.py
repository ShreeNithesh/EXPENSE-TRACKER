import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor


def test_preprocess_and_features():
    df = pd.DataFrame({
        'merchant': ['Starbucks', 'Walmart', 'Netflix'],
        'category': ['food', 'shopping', 'entertainment'],
        'amt': [5.2, 123.0, 12.99]
    })
    pre = DataPreprocessor()
    df_proc, le = pre.preprocess_data(df)
    X, y, vec, scaler, tokenizer, sequences = pre.create_features(df_proc)
    assert X.shape[0] == 3
    assert len(y) == 3
    assert hasattr(vec, 'transform')
    assert hasattr(scaler, 'transform')


def test_small_sample_sampling():
    df = pd.DataFrame({'merchant': ['a']*5, 'category': ['x']*5, 'amt': list(range(5))})
    pre = DataPreprocessor()
    df_proc, le = pre.preprocess_data(df)
    assert len(df_proc) == 5
