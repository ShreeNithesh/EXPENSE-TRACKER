import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer


def test_train_models_runs():
    df = pd.DataFrame({
        'merchant': ['Starbucks', 'Walmart', 'Netflix', 'Amazon'],
        'category': ['food', 'shopping', 'entertainment', 'shopping'],
        'amt': [5.2, 123.0, 12.99, 45.0]
    })
    pre = DataPreprocessor()
    df_proc, le = pre.preprocess_data(df)
    X, y, vec, scaler, tokenizer, sequences = pre.create_features(df_proc)
    trainer = ModelTrainer()
    model = trainer.train_models(X, y, le.classes_, tokenizer=tokenizer, sequences=sequences)
    # ensure it returns a model or sentinel
    assert model is not None
