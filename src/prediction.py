import numpy as np
import pickle as pkl
import json
import glob
import os
from src.utils import clean_text
import scipy.sparse as sparse

# Optional TensorFlow imports
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

def load_artifacts():
    """Load saved model, vectorizer, scaler, encoder."""
    # load metadata to know which model to pick
    metadata_path = 'models/metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = None

    # Load vectorizer, scaler, label encoder (look for latest timestamped files)
    vectorizer = None
    scaler = None
    label_encoder = None
    # tfidf vectorizer
    for p in sorted(glob.glob('models/*tfidf*'), reverse=True):
        try:
            with open(p, 'rb') as f:
                vectorizer = pkl.load(f)
            break
        except Exception:
            continue
    for p in sorted(glob.glob('models/*scaler*'), reverse=True):
        try:
            with open(p, 'rb') as f:
                scaler = pkl.load(f)
            break
        except Exception:
            continue
    for p in sorted(glob.glob('models/*label_encoder*'), reverse=True):
        try:
            with open(p, 'rb') as f:
                label_encoder = pkl.load(f)
            break
        except Exception:
            continue

    # load model
    model = None
    rnn_model = None
    tokenizer = None
    if metadata and metadata.get('best_model') == 'rnn' and load_model is not None:
        # prefer rnn model
        if os.path.exists('models/rnn_model.h5'):
            rnn_model = load_model('models/rnn_model.h5')
        if os.path.exists('models/tokenizer.pkl'):
            with open('models/tokenizer.pkl', 'rb') as f:
                tokenizer = pkl.load(f)
        model = rnn_model
    else:
        # fallback: find most recent classical model file
        models = sorted(glob.glob('models/model_*.pkl'), reverse=True)
        if models:
            with open(models[0], 'rb') as f:
                model = pkl.load(f)

    return model, vectorizer, scaler, label_encoder, tokenizer

def predict_category(merchant, amount):
    """Predict category for a single transaction."""
    model, vectorizer, scaler, label_encoder, tokenizer = load_artifacts()
    # Basic validation of required artifacts
    if model is None or vectorizer is None or scaler is None or label_encoder is None:
        raise RuntimeError("Model artifacts not found. Please run training to generate model/vectorizer/scaler/label_encoder files in the 'models' directory.")
    merchant_clean = clean_text(merchant)
    amt_log = np.log1p(float(amount))
    tfidf = vectorizer.transform([merchant_clean])
    # Ensure numeric features are 2D numpy arrays
    numerical = scaler.transform(np.array([[amt_log]]))
    numerical = np.asarray(numerical)

    # Handle sparse TF-IDF output (from scikit-learn) vs dense arrays and ensure 2D
    try:
        if sparse.issparse(tfidf):
            tfidf_arr = tfidf.toarray()
        else:
            tfidf_arr = np.asarray(tfidf)
    except Exception:
        tfidf_arr = np.asarray(tfidf)

    # Ensure both are 2D: reshape 1D arrays to (1, -1)
    if tfidf_arr.ndim == 1:
        tfidf_arr = tfidf_arr.reshape(1, -1)
    if numerical.ndim == 1:
        numerical = numerical.reshape(1, -1)

    # As a last resort, if dimensions still differ, try converting sparse to dense then reshape
    try:
        X = np.hstack([tfidf_arr, numerical])
    except Exception as e:
        # Write debug details to a file for inspection
        debug_path = 'models/predict_debug.json'
        info = {
            'tfidf_shape': getattr(tfidf_arr, 'shape', None),
            'tfidf_ndim': getattr(tfidf_arr, 'ndim', None),
            'numerical_shape': getattr(numerical, 'shape', None),
            'numerical_ndim': getattr(numerical, 'ndim', None),
            'tfidf_type': type(tfidf_arr).__name__,
            'numerical_type': type(numerical).__name__,
        }
        try:
            with open(debug_path, 'w', encoding='utf-8') as df:
                json.dump(info, df)
        except Exception:
            pass
        # Re-raise the original exception so the API returns a 500 and we can see traceback
        raise
    # If model is a Keras RNN (loaded via load_model), expect sequences
    if hasattr(model, 'predict') and tokenizer is not None and hasattr(model, 'predict') and hasattr(tokenizer, 'texts_to_sequences'):
        # build sequences
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = tokenizer.texts_to_sequences([merchant_clean])
            seq = pad_sequences(seq, maxlen=20, padding='post', truncating='post')
            preds = model.predict(seq)
            pred_encoded = int(np.argmax(preds, axis=1)[0])
            category = label_encoder.inverse_transform([pred_encoded])[0]
            confidence = float(np.max(preds))
            return category, confidence
        except Exception:
            pass

    # classical model path
    pred_encoded = model.predict(X)[0]
    try:
        category = label_encoder.inverse_transform([pred_encoded])[0]
    except Exception:
        category = str(pred_encoded)
    proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [1.0]
    confidence = np.max(proba)
    return category, confidence