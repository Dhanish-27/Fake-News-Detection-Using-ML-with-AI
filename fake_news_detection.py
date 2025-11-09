"""
Fake News Detection using Machine Learning and NLP
=================================================

This project implements fake news detection using three different machine learning models:
1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine (SVM)

The project uses TF-IDF vectorization for text preprocessing and provides comprehensive
evaluation metrics and visualizations.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import string
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FakeNewsDetector:
    """
    A comprehensive fake news detection system using multiple ML models.
    """
    
    def __init__(self):
        """Initialize the fake news detector."""
        self.true_data = None
        self.fake_data = None
        self.combined_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.results = {}
        
    def load_data(self, true_path='NLP_Analysis/True.csv', fake_path='NLP_Analysis/Fake.csv'):
        """
        Load and combine the true and fake news datasets.
        
        Args:
            true_path (str): Path to the true news CSV file
            fake_path (str): Path to the fake news CSV file
        """
        print("Loading datasets...")
        
        # Load datasets
        self.true_data = pd.read_csv(true_path)
        self.fake_data = pd.read_csv(fake_path)
        
        # Add labels
        self.true_data['label'] = 1  # 1 for real news
        self.fake_data['label'] = 0  # 0 for fake news
        
        # Combine datasets
        self.combined_data = pd.concat([self.true_data, self.fake_data], ignore_index=True)
        
        # Shuffle the data
        self.combined_data = self.combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dataset loaded successfully!")
        print(f"True news articles: {len(self.true_data)}")
        print(f"Fake news articles: {len(self.fake_data)}")
        print(f"Total articles: {len(self.combined_data)}")
        print(f"Dataset shape: {self.combined_data.shape}")
        
    def preprocess_text(self, text):
        """
        Preprocess text data by removing special characters, converting to lowercase,
        and removing extra whitespaces.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self):
        """
        Prepare the data for training by preprocessing text and creating features.
        """
        print("\nPreparing data for training...")
        
        # Combine title and text for better feature extraction
        self.combined_data['combined_text'] = (
            self.combined_data['title'].fillna('') + ' ' + 
            self.combined_data['text'].fillna('')
        )
        
        # Preprocess the combined text
        self.combined_data['processed_text'] = self.combined_data['combined_text'].apply(
            self.preprocess_text
        )
        
        # Remove empty texts
        self.combined_data = self.combined_data[
            self.combined_data['processed_text'].str.len() > 0
        ]
        
        # Create feature matrix using TF-IDF
        print("Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(self.combined_data['processed_text'])
        y = self.combined_data['label']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def train_models(self):
        """
        Train the three machine learning models.
        """
        print("\nTraining machine learning models...")
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'SVM': SVC(
                random_state=42,
                kernel='linear',
                C=1.0,
                probability=True
            )
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    def evaluate_models(self):
        """
        Evaluate and compare the performance of all models.
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Detailed classification reports
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*60)
        
        for name, results in self.results.items():
            print(f"\n{name.upper()}:")
            print("-" * 40)
            print(classification_report(
                self.y_test, 
                results['predictions'],
                target_names=['Fake News', 'Real News']
            ))
    
    def plot_results(self):
        """
        Create visualizations for model comparison and performance analysis.
        """
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        precisions = [self.results[model]['precision'] for model in models]
        recalls = [self.results[model]['recall'] for model in models]
        
        axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0.85, 1.0)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
        
        axes[0, 1].bar(models, f1_scores, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0.85, 1.0)
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Precision vs Recall
        axes[1, 0].scatter(precisions, recalls, s=100, c=['red', 'green', 'blue'])
        for i, model in enumerate(models):
            axes[1, 0].annotate(model, (precisions[i], recalls[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics radar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(models):
            values = [
                self.results[model]['accuracy'],
                self.results[model]['precision'],
                self.results[model]['recall'],
                self.results[model]['f1_score']
            ]
            values += values[:1]  # Complete the circle
            
            axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=model)
            axes[1, 1].fill(angles, values, alpha=0.25)
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim(0.85, 1.0)
        axes[1, 1].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Fake News', 'Real News'],
                       yticklabels=['Fake News', 'Real News'],
                       ax=axes[i])
            axes[i].set_title(f'{name}\nConfusion Matrix', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_news(self, text, model_name='Logistic Regression'):
        """
        Predict whether a given text is fake or real news.
        
        Args:
            text (str): News text to classify
            model_name (str): Name of the model to use for prediction
            
        Returns:
            dict: Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        # Preprocess the input text
        processed_text = self.preprocess_text(text)
        
        # Transform using the trained vectorizer
        text_vector = self.vectorizer.transform([processed_text])
        
        # Make prediction
        model = self.results[model_name]['model']
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        result = {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'processed_text': processed_text[:200] + "..." if len(processed_text) > 200 else processed_text,
            'prediction': 'Real News' if prediction == 1 else 'Fake News',
            'confidence': {
                'Fake News': probability[0],
                'Real News': probability[1]
            },
            'model_used': model_name
        }
        
        return result
    
    def save_models(self, filename_prefix='fake_news_model'):
        """
        Save the trained models and vectorizer to disk.
        
        Args:
            filename_prefix (str): Prefix for the saved files
        """
        print(f"\nSaving models to disk...")
        
        # Save vectorizer
        with open(f'{filename_prefix}_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save models
        for name, results in self.results.items():
            model_filename = f'{filename_prefix}_{name.lower().replace(" ", "_")}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(results['model'], f)
        
        print("Models saved successfully!")
    
    def load_models(self, filename_prefix='fake_news_model'):
        """
        Load previously saved models and vectorizer from disk.
        
        Args:
            filename_prefix (str): Prefix for the saved files
        """
        print(f"Loading models from disk...")
        
        # Load vectorizer
        with open(f'{filename_prefix}_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load models
        model_files = {
            'Logistic Regression': f'{filename_prefix}_logistic_regression.pkl',
            'Naive Bayes': f'{filename_prefix}_naive_bayes.pkl',
            'SVM': f'{filename_prefix}_svm.pkl'
        }
        
        self.models = {}
        for name, filename in model_files.items():
            try:
                with open(filename, 'rb') as f:
                    self.models[name] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: Model file {filename} not found.")
        
        print("Models loaded successfully!")


def main():
    """
    Main function to run the fake news detection pipeline.
    """
    print("="*80)
    print("FAKE NEWS DETECTION USING MACHINE LEARNING")
    print("="*80)
    
    # Initialize the detector
    detector = FakeNewsDetector()
    
    # Load and prepare data
    detector.load_data()
    detector.prepare_data()
    
    # Train models
    detector.train_models()
    
    # Evaluate models
    detector.evaluate_models()
    
    # Create visualizations
    detector.plot_results()
    
    # Save models
    detector.save_models()
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Sample texts for testing
    sample_texts = [
        "Scientists have discovered a new planet that could support life. The planet is located 100 light years away and has similar conditions to Earth.",
        "BREAKING: Local man discovers secret government conspiracy involving aliens and time travel. Officials deny everything but sources say otherwise.",
        "The stock market reached record highs today as investors showed confidence in the economy. Technology stocks led the gains with significant increases.",
        "SHOCKING: Celebrities are secretly controlling the weather using HAARP technology. This explains all the recent climate changes!"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nExample {i}:")
        print("-" * 40)
        result = detector.predict_news(text, 'Logistic Regression')
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']['Real News']:.2%} Real, {result['confidence']['Fake News']:.2%} Fake")
    
    print("\n" + "="*80)
    print("FAKE NEWS DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nFiles generated:")
    print("- model_performance_comparison.png")
    print("- confusion_matrices.png")
    print("- fake_news_model_vectorizer.pkl")
    print("- fake_news_model_logistic_regression.pkl")
    print("- fake_news_model_naive_bayes.pkl")
    print("- fake_news_model_svm.pkl")


if __name__ == "__main__":
    main()
