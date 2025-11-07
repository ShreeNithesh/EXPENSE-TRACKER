import numpy as np
# Lazy import plotting libraries only when needed to avoid import-time failures
import pickle as pkl
import os
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import pinv

# Optional TensorFlow imports for RNN/LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    tf = None
    Sequential = None
    Embedding = None
    LSTM = None
    Dense = None
    TF_AVAILABLE = False

class LogisticRegression:
    """Wrapper around scikit-learn LogisticRegression for compatibility."""
    def __init__(self):
        self.model = SKLogReg(max_iter=2000)

    def fit(self, X, y):
        # simple grid search for C
        param_grid = {'C': [0.01, 0.1, 1.0]}
        gs = GridSearchCV(self.model, param_grid, cv=3, n_jobs=1)
        try:
            gs.fit(X, y)
            self.model = gs.best_estimator_
        except ValueError:
            # dataset too small for cross-validation; fallback to direct fit
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class KNN:
    def __init__(self):
        self.model = KNeighborsClassifier()

    def fit(self, X, y):
        param_grid = {'n_neighbors': [3, 5, 7]}
        gs = GridSearchCV(self.model, param_grid, cv=3, n_jobs=1)
        try:
            gs.fit(X, y)
            self.model = gs.best_estimator_
        except ValueError:
            # dataset too small for cross-validation; fallback to direct fit
            self.model.fit(X, y)
        # ensure n_neighbors does not exceed fitted sample size (handles tiny datasets)
        try:
            n_fit = getattr(self.model, 'n_samples_fit_', None)
            if n_fit is not None and hasattr(self.model, 'n_neighbors'):
                if self.model.n_neighbors > n_fit:
                    self.model.n_neighbors = max(1, n_fit)
        except Exception:
            pass

    def predict(self, X):
        return self.model.predict(X)


class ELM:
    """A lightweight Extreme Learning Machine implementation.

    Single hidden layer feedforward network with random input weights and
    closed-form solution for output weights (ridge regression).
    """
    def __init__(self, n_hidden=1000, activation='tanh', reg=1e-3, random_state=42):
        self.n_hidden = n_hidden
        self.activation = activation
        self.reg = reg
        self.random_state = random_state
        self._is_fitted = False

    def _activate(self, X):
        if self.activation == 'tanh':
            return np.tanh(X)
        if self.activation == 'relu':
            return np.maximum(0, X)
        # default linear
        return X

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        # random input weights and biases
        self.W = rng.normal(size=(n_features, self.n_hidden))
        self.b = rng.normal(size=(self.n_hidden,))
        H = self._activate(X.dot(self.W) + self.b)
        # one-hot encode targets
        y = np.asarray(y).reshape(-1, 1)
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        T = ohe.fit_transform(y)
        self._ohe = ohe
        # ridge solution: beta = (H^T H + reg*I)^-1 H^T T
        # use pinv for numerical stability on small datasets
        n_hidden = H.shape[1]
        I = np.eye(n_hidden)
        try:
            self.beta = np.linalg.inv(H.T.dot(H) + self.reg * I).dot(H.T).dot(T)
        except Exception:
            self.beta = pinv(H.T.dot(H) + self.reg * I).dot(H.T).dot(T)
        self._is_fitted = True

    def predict(self, X):
        if not self._is_fitted:
            raise ValueError('ELM not fitted')
        H = self._activate(X.dot(self.W) + self.b)
        Y = H.dot(self.beta)
        preds = np.argmax(Y, axis=1)
        return preds

    def predict_proba(self, X):
        if not self._is_fitted:
            raise ValueError('ELM not fitted')
        H = self._activate(X.dot(self.W) + self.b)
        Y = H.dot(self.beta)
        # softmax
        exp = np.exp(Y - np.max(Y, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def classification_report(y_true, y_pred, classes):
    report = ""
    for cls in classes:
        idx = (y_true == cls)
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        report += f"Class {cls}: Precision {precision:.4f}, Recall {recall:.4f}, F1 {f1:.4f}\n"
    return report

class ModelTrainer:
    def train_test_split(self, X, y, test_size=0.2, return_indices=False):
        """Simple train/test split that is robust for small datasets.

        When return_indices=True this also returns the train/test indices so
        callers can apply the same split to other arrays (e.g. token sequences).
        """
        n = len(y)
        if n == 0:
            raise ValueError("Empty dataset")
        idx = np.random.permutation(n)
        test_n = int(n * test_size)
        # ensure at least one sample in test when possible
        if test_n < 1 and n > 1:
            test_n = 1
        # ensure at least one sample in train when possible
        if test_n >= n:
            test_n = n - 1 if n > 1 else 0
        test_idx = idx[:test_n]
        train_idx = idx[test_n:]
        if return_indices:
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def train_models(self, X, y, classes, tokenizer=None, sequences=None, force_model=None, rnn_epochs=3):
        """Train available models.

        Args:
            X: 2D feature matrix (tfidf + numerical)
            y: labels
            classes: array-like of class names
            tokenizer: optional Keras Tokenizer object (if sequences provided)
            sequences: optional padded token sequences for RNN input
            force_model: one of None|'lr'|'knn'|'rnn' to force training/saving a particular model
            rnn_epochs: epochs for RNN training when used
        """
        # split and keep indices so we can align sequences if provided
        X_train, X_test, y_train, y_test, train_idx, test_idx = self.train_test_split(
            X, y, test_size=0.2, return_indices=True
        )

        # Train logistic regression if requested
        acc_lr = -1.0
        if force_model in (None, 'lr'):
            lr = LogisticRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            acc_lr = accuracy_score(y_test, y_pred_lr)
        else:
            lr = None
        print("Logistic Regression Results:")
        print(f"Accuracy: {acc_lr:.4f}")
        if lr is not None:
            try:
                print(classification_report(y_test, y_pred_lr, range(len(classes))))
            except Exception:
                pass

        # Train KNN if requested
        acc_knn = -1.0
        if force_model in (None, 'knn'):
            knn = KNN()
            knn.fit(X_train, y_train)
            y_pred_knn = knn.predict(X_test)
            acc_knn = accuracy_score(y_test, y_pred_knn)
        else:
            knn = None
        print("\nKNN Results:")
        print(f"Accuracy: {acc_knn:.4f}")
        if knn is not None:
            try:
                print(classification_report(y_test, y_pred_knn, range(len(classes))))
            except Exception:
                pass

        # Train ELM (Extreme Learning Machine) - lightweight and fast
        acc_elm = -1.0
        elm = None
        if force_model in (None, 'elm'):
            try:
                elm = ELM(n_hidden=500, activation='tanh', reg=1e-3)
                elm.fit(X_train, y_train)
                y_pred_elm = elm.predict(X_test)
                acc_elm = accuracy_score(y_test, y_pred_elm)
                print('\nELM Results:')
                print(f'Accuracy: {acc_elm:.4f}')
                try:
                    print(classification_report(y_test, y_pred_elm, range(len(classes))))
                except Exception:
                    pass
            except Exception:
                elm = None

        # Confusion matrix for better model (LR) - plotting done lazily
        try:
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for true, pred in zip(y_test, y_pred_lr):
                cm[true, pred] += 1
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix - Logistic Regression')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.savefig('confusion_matrix.png')
                plt.close()
            except Exception:
                pass
        except Exception:
            pass

        # If sequences provided and TF available, train an RNN and compare
        rnn_model = None
        acc_rnn = -1.0
        if sequences is not None and TF_AVAILABLE and force_model in (None, 'rnn'):
            # align sequence split with the same indices used for X/y
            try:
                seq_train = sequences[train_idx]
                seq_test = sequences[test_idx]
            except Exception:
                # fallback: try to split sequences independently
                seq_train, seq_test = self.train_test_split(sequences, y)[:2]

            # Build a small LSTM
            if tokenizer is not None:
                num_words = getattr(tokenizer, 'num_words', None)
                if num_words:
                    vocab_size = min(5000, int(num_words))
                else:
                    # fallback to word_index size
                    try:
                        vocab_size = min(5000, len(tokenizer.word_index) + 1)
                    except Exception:
                        vocab_size = 5000
            else:
                vocab_size = 5000

            max_len = seq_train.shape[1]
            num_classes = len(np.unique(y))
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
                LSTM(64),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            # small training for speed; increase epochs for real training
            model.fit(seq_train, y_train, validation_data=(seq_test, y_test), epochs=rnn_epochs, batch_size=64, verbose=1)
            loss, acc = model.evaluate(seq_test, y_test, verbose=0)
            acc_rnn = acc
            rnn_model = model
            print('\nRNN (LSTM) Results:')
            print(f'Accuracy: {acc_rnn:.4f}')

        # Choose best among available models
        best_model = None
        best_acc = -1.0
        best_name = None
        # choose best among those actually trained
        if acc_lr > best_acc and lr is not None:
            best_model = lr
            best_acc = acc_lr
            best_name = 'lr'
        if acc_knn > best_acc and knn is not None:
            best_model = knn
            best_acc = acc_knn
            best_name = 'knn'
        # ELM support: if we trained an elm variable include it
        try:
            if 'elm' in locals() and elm is not None:
                acc_elm = locals().get('acc_elm', -1.0)
                if acc_elm > best_acc:
                    best_model = elm
                    best_acc = acc_elm
                    best_name = 'elm'
        except Exception:
            pass
        if acc_rnn > best_acc and rnn_model is not None:
            # Save Keras model separately and mark as best
            os.makedirs('models', exist_ok=True)
            rnn_model.save('models/rnn_model.h5')
            if tokenizer is not None:
                with open('models/tokenizer.pkl', 'wb') as f:
                    pkl.dump(tokenizer, f)
            best_model = 'rnn'  # sentinel to indicate rnn file is best
            best_acc = acc_rnn
            best_name = 'rnn'

        # Save classical model object for non-RNN best with versioning
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        registry_entry = {
            'timestamp': timestamp,
            'best_name': None,
            'files': []
        }
        if best_model != 'rnn':
            model_path = f'models/model_{best_name}_{timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pkl.dump(best_model, f)
            registry_entry['best_name'] = best_name
            registry_entry['files'].append(model_path)
        else:
            registry_entry['best_name'] = 'rnn'
            # rnn file already saved as models/rnn_model.h5 earlier
            registry_entry['files'].append('models/rnn_model.h5')
            if tokenizer is not None:
                registry_entry['files'].append('models/tokenizer.pkl')

        # Save metadata for consumers (which model, accuracies)
        metadata = {
            'best_model': best_name if best_model is not None else None,
            'accuracy': {
                'lr': None if acc_lr == -1.0 else float(acc_lr),
                'knn': None if acc_knn == -1.0 else float(acc_knn),
                'elm': None if acc_elm == -1.0 else float(acc_elm),
                'rnn': None if acc_rnn == -1.0 else float(acc_rnn)
            },
            'classes': list(classes)
        }
        with open('models/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return best_model