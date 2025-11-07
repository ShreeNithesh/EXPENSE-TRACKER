"""
Model Evaluation and Benchmarking System
Provides comprehensive evaluation metrics and model comparison
"""
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer, LogisticRegression, KNN, ELM
import pickle as pkl

# Optional TensorFlow imports for RNN evaluation
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    tf = None
    load_model = None
    TF_AVAILABLE = False

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.benchmark_model = None
        self.benchmark_metrics = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name, label_encoder):
        """Comprehensive evaluation of a single model"""
        try:
            # Make predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                return None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Per-class metrics
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, labels=range(len(label_encoder.classes_))
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(
                y_test, y_pred, 
                target_names=label_encoder.classes_, 
                output_dict=True
            )
            
            metrics = {
                'model_name': model_name,
                'accuracy': float(accuracy),
                'precision_weighted': float(precision),
                'recall_weighted': float(recall),
                'f1_weighted': float(f1),
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'f1_per_class': f1_per_class.tolist(),
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'support_per_class': support.tolist() if hasattr(support, 'tolist') else support,
                'total_samples': len(y_test),
                'num_classes': len(label_encoder.classes_),
                'class_names': label_encoder.classes_.tolist()
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return None
    
    def evaluate_rnn_model(self, model_path, tokenizer_path, X_test, y_test, sequences_test, label_encoder):
        """Evaluate RNN model specifically"""
        if not TF_AVAILABLE or not os.path.exists(model_path):
            return None
            
        try:
            # Load RNN model and tokenizer
            rnn_model = load_model(model_path)
            
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pkl.load(f)
            else:
                return None
            
            # Make predictions using sequences
            if sequences_test is not None:
                predictions = rnn_model.predict(sequences_test)
                y_pred = np.argmax(predictions, axis=1)
                
                return self.evaluate_model(rnn_model, None, y_test, 'RNN', label_encoder)
            
        except Exception as e:
            print(f"Error evaluating RNN: {str(e)}")
            return None
    
    def benchmark_all_models(self, data_path='data/improved_transactions.csv'):
        """Comprehensive benchmarking of all available models"""
        print("🔄 Starting comprehensive model benchmarking...")
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(data_path)
        df_processed, label_encoder = preprocessor.preprocess_data(df)
        res = preprocessor.create_features(df_processed)
        
        if len(res) == 4:
            X, y, vectorizer, scaler = res
            tokenizer, sequences = None, None
        else:
            X, y, vectorizer, scaler, tokenizer, sequences = res
        
        # Split data
        trainer = ModelTrainer()
        if sequences is not None:
            X_train, X_test, y_train, y_test, train_idx, test_idx = trainer.train_test_split(
                X, y, test_size=0.2, return_indices=True
            )
            seq_train = sequences[train_idx]
            seq_test = sequences[test_idx]
        else:
            X_train, X_test, y_train, y_test = trainer.train_test_split(X, y, test_size=0.2)
            seq_train, seq_test = None, None
        
        results = {}
        
        # Evaluate Logistic Regression
        print("📊 Evaluating Logistic Regression...")
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        lr_metrics = self.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression', label_encoder)
        if lr_metrics:
            results['logistic_regression'] = lr_metrics
        
        # Evaluate KNN
        print("📊 Evaluating K-Nearest Neighbors...")
        knn_model = KNN()
        knn_model.fit(X_train, y_train)
        knn_metrics = self.evaluate_model(knn_model, X_test, y_test, 'K-Nearest Neighbors', label_encoder)
        if knn_metrics:
            results['knn'] = knn_metrics
        
        # Evaluate ELM
        print("📊 Evaluating Extreme Learning Machine...")
        try:
            elm_model = ELM(n_hidden=500, activation='tanh', reg=1e-3)
            elm_model.fit(X_train, y_train)
            elm_metrics = self.evaluate_model(elm_model, X_test, y_test, 'Extreme Learning Machine', label_encoder)
            if elm_metrics:
                results['elm'] = elm_metrics
        except Exception as e:
            print(f"ELM evaluation failed: {str(e)}")
        
        # Evaluate RNN if available
        if TF_AVAILABLE and sequences is not None:
            print("📊 Training and evaluating RNN...")
            try:
                # Train RNN for evaluation
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Embedding, LSTM, Dense
                
                vocab_size = min(5000, len(tokenizer.word_index) + 1) if tokenizer else 5000
                max_len = seq_train.shape[1] if seq_train is not None else 20
                num_classes = len(np.unique(y))
                
                rnn_model = Sequential([
                    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
                    LSTM(64),
                    Dense(num_classes, activation='softmax')
                ])
                rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # Train with validation
                history = rnn_model.fit(
                    seq_train, y_train, 
                    validation_data=(seq_test, y_test), 
                    epochs=5, 
                    batch_size=64, 
                    verbose=1
                )
                
                # Evaluate RNN
                predictions = rnn_model.predict(seq_test)
                y_pred_rnn = np.argmax(predictions, axis=1)
                
                # Calculate RNN metrics manually since we need custom evaluation
                accuracy = accuracy_score(y_test, y_pred_rnn)
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_rnn, average='weighted')
                
                rnn_metrics = {
                    'model_name': 'RNN (LSTM)',
                    'accuracy': float(accuracy),
                    'precision_weighted': float(precision),
                    'recall_weighted': float(recall),
                    'f1_weighted': float(f1),
                    'total_samples': len(y_test),
                    'num_classes': len(label_encoder.classes_),
                    'class_names': label_encoder.classes_.tolist(),
                    'training_history': {
                        'final_loss': float(history.history['loss'][-1]),
                        'final_val_loss': float(history.history['val_loss'][-1]),
                        'final_accuracy': float(history.history['accuracy'][-1]),
                        'final_val_accuracy': float(history.history['val_accuracy'][-1])
                    }
                }
                results['rnn'] = rnn_metrics
                
                # Save RNN model for later use
                os.makedirs('models', exist_ok=True)
                rnn_model.save('models/rnn_model_benchmark.h5')
                
            except Exception as e:
                print(f"RNN evaluation failed: {str(e)}")
        
        # Determine benchmark model (best F1 score)
        best_model = None
        best_f1 = -1
        
        for model_name, metrics in results.items():
            if metrics and metrics.get('f1_weighted', 0) > best_f1:
                best_f1 = metrics['f1_weighted']
                best_model = model_name
        
        # Create comprehensive benchmark report
        benchmark_report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(df),
                'features': X.shape[1],
                'classes': len(label_encoder.classes_),
                'class_names': label_encoder.classes_.tolist(),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'model_results': results,
            'benchmark_model': best_model,
            'benchmark_metrics': results.get(best_model, {}) if best_model else {},
            'model_comparison': self._create_comparison_table(results)
        }
        
        # Save benchmark report
        os.makedirs('models', exist_ok=True)
        with open('models/benchmark_report.json', 'w', encoding='utf-8') as f:
            json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Benchmarking complete! Best model: {best_model} (F1: {best_f1:.4f})")
        return benchmark_report
    
    def _create_comparison_table(self, results):
        """Create a comparison table of all models"""
        comparison = []
        
        for model_name, metrics in results.items():
            if metrics:
                comparison.append({
                    'model': metrics['model_name'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision_weighted'],
                    'recall': metrics['recall_weighted'],
                    'f1_score': metrics['f1_weighted'],
                    'samples': metrics['total_samples']
                })
        
        # Sort by F1 score descending
        comparison.sort(key=lambda x: x['f1_score'], reverse=True)
        return comparison
    
    def load_benchmark_report(self):
        """Load existing benchmark report"""
        report_path = 'models/benchmark_report.json'
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_model_summary(self):
        """Get a summary of model performance"""
        report = self.load_benchmark_report()
        if not report:
            return "No benchmark report found. Run benchmarking first."
        
        summary = f"""
📊 MODEL BENCHMARK SUMMARY
========================
🏆 Best Model: {report['benchmark_model']}
📈 Accuracy: {report['benchmark_metrics'].get('accuracy', 0):.4f}
📈 F1-Score: {report['benchmark_metrics'].get('f1_weighted', 0):.4f}
📈 Precision: {report['benchmark_metrics'].get('precision_weighted', 0):.4f}
📈 Recall: {report['benchmark_metrics'].get('recall_weighted', 0):.4f}

📋 All Models Comparison:
"""
        
        for model in report['model_comparison']:
            summary += f"\n{model['model']}: F1={model['f1_score']:.4f}, Acc={model['accuracy']:.4f}"
        
        return summary

def run_comprehensive_benchmark():
    """Run complete model benchmarking"""
    evaluator = ModelEvaluator()
    return evaluator.benchmark_all_models()

if __name__ == "__main__":
    # Run benchmarking when script is executed directly
    run_comprehensive_benchmark()