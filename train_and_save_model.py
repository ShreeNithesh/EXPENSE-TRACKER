"""Main script to train and save the model."""
import os
import pickle as pkl
import argparse
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from datetime import datetime
import json

os.makedirs('models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['lr', 'knn', 'rnn'], default=None, help='Force which model to train/save')
parser.add_argument('--epochs', type=int, default=3, help='RNN epochs')
parser.add_argument('--test-size', type=float, default=0.2, help='Test split fraction')
args = parser.parse_args()

preprocessor = DataPreprocessor()
# Use improved dataset with realistic merchant names
df = preprocessor.load_data('data/improved_transactions.csv')
df_processed, label_encoder = preprocessor.preprocess_data(df)
res = preprocessor.create_features(df_processed)
# backward compatible unpack
if len(res) == 4:
    X, y, vectorizer, scaler = res
    tokenizer, sequences = None, None
else:
    X, y, vectorizer, scaler, tokenizer, sequences = res

trainer = ModelTrainer()
# Pass tokenizer and sequences for optional RNN training if available
model = trainer.train_models(X, y, label_encoder.classes_, tokenizer=tokenizer, sequences=sequences, force_model=args.model, rnn_epochs=args.epochs)

# Run comprehensive benchmarking to get detailed metrics
print("\n🔄 Running comprehensive model evaluation...")
from src.model_evaluation import ModelEvaluator
evaluator = ModelEvaluator()
benchmark_report = evaluator.benchmark_all_models()

# Save preprocessing artifacts with timestamped names and register
timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
vec_path = f'models/tfidf_vectorizer_{timestamp}.pkl'
scaler_path = f'models/scaler_{timestamp}.pkl'
le_path = f'models/label_encoder_{timestamp}.pkl'
with open(vec_path, 'wb') as f:
    pkl.dump(vectorizer, f)
with open(scaler_path, 'wb') as f:
    pkl.dump(scaler, f)
with open(le_path, 'wb') as f:
    pkl.dump(label_encoder, f)

# Append to registry
registry_path = 'models/registry.json'
entry = {
    'timestamp': timestamp,
    'vectorizer': vec_path,
    'scaler': scaler_path,
    'label_encoder': le_path
}
registry = []
if os.path.exists(registry_path):
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
    except Exception:
        registry = []
registry.append(entry)
with open(registry_path, 'w', encoding='utf-8') as f:
    json.dump(registry, f, indent=2)

print("Training complete! Model saved.")
print("Unique categories:", label_encoder.classes_)