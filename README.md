# Fake News Detection using Machine Learning and NLP

This project implements a comprehensive fake news detection system using three different machine learning algorithms: **Logistic Regression**, **Naive Bayes**, and **Support Vector Machine (SVM)**. The system uses Natural Language Processing (NLP) techniques to analyze news articles and classify them as either real or fake.

## 🎯 Project Overview

With the rapid spread of misinformation online, this tool aims to help identify fake news articles using machine learning. The project analyzes text content using TF-IDF vectorization and trains multiple models to achieve high accuracy in fake news detection.

## 📊 Dataset

The project uses two CSV files:
- **True.csv**: Contains real news articles (labeled as 1)
- **Fake.csv**: Contains fake news articles (labeled as 0)

The datasets include the following columns:
- `title`: News article headline
- `text`: Full article content
- `subject`: News category
- `date`: Publication date

## 🚀 Features

- **🌐 Web Application**: Beautiful, responsive web interface built with Flask
- **🤖 Multiple ML Models**: Logistic Regression, Naive Bayes, and SVM
- **🔧 Advanced Text Preprocessing**: URL removal, special character cleaning, TF-IDF vectorization
- **📊 Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score metrics
- **📈 Visualizations**: Performance comparison charts and confusion matrices
- **🎯 Real-time Predictions**: Instant analysis with confidence scores
- **📱 Mobile Responsive**: Works perfectly on desktop, tablet, and mobile devices
- **💾 Model Persistence**: Save and load trained models
- **🔌 API Endpoints**: RESTful API for integration with other applications

## 📁 Project Structure

```
├── fake_news_detection.py    # Main ML implementation
├── app.py                   # Flask web application
├── run_web_app.py          # Web app startup script
├── requirements.txt         # Project dependencies
├── README.md               # Project documentation
├── templates/              # HTML templates for web app
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   ├── train.html
│   └── error.html
├── static/                 # CSS and JavaScript files
│   ├── style.css
│   └── script.js
├── NLP_Analysis/
│   ├── True.csv           # Real news dataset
│   └── Fake.csv           # Fake news dataset
└── Generated Files:
    ├── model_performance_comparison.png
    ├── confusion_matrices.png
    └── *.pkl files (saved models)
```

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your datasets are in the correct location:**
   - Place `True.csv` and `Fake.csv` in the `NLP_Analysis/` folder

## 🎮 Usage

### Web Application (Recommended)

Start the Flask web application for an easy-to-use interface:

```bash
# Option 1: Using the startup script (recommended)
python run_web_app.py

# Option 2: Direct Flask command
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

The web app provides:
- 🖥️ **User-friendly interface** for analyzing news articles
- 🎛️ **Model selection** between Logistic Regression, Naive Bayes, and SVM
- 📊 **Real-time predictions** with confidence scores
- 📈 **Visual results** with progress bars and detailed analysis
- 📱 **Responsive design** that works on desktop and mobile

### Command Line Usage

Run the complete fake news detection pipeline:

```bash
python fake_news_detection.py
```

This will:
- Load and preprocess the datasets
- Train all three models
- Evaluate and compare performance
- Generate visualizations
- Save trained models
- Show example predictions

### Programmatic Usage

```python
from fake_news_detection import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Load and prepare data
detector.load_data()
detector.prepare_data()

# Train models
detector.train_models()

# Evaluate models
detector.evaluate_models()

# Predict new articles
result = detector.predict_news("Your news article text here")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Loading Pre-trained Models

```python
# Load previously saved models
detector = FakeNewsDetector()
detector.load_models()

# Make predictions with loaded models
result = detector.predict_news("Article text", model_name='Logistic Regression')
```

## 📈 Model Performance

The project trains and compares three models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | ~96% | ~96% | ~96% | ~96% |
| **Naive Bayes** | ~93% | ~93% | ~93% | ~93% |
| **SVM** | ~95% | ~95% | ~95% | ~95% |

*Note: Actual performance may vary based on dataset characteristics*

## 🔧 Technical Details

### Text Preprocessing Pipeline

1. **Text Cleaning**: Remove URLs, emails, special characters
2. **Normalization**: Convert to lowercase, remove extra whitespaces
3. **Feature Extraction**: TF-IDF vectorization with 5000 features
4. **N-gram Analysis**: Uses unigrams and bigrams for better context

### Model Configurations

- **Logistic Regression**: C=1.0, max_iter=1000
- **Naive Bayes**: Multinomial with alpha=0.1
- **SVM**: Linear kernel, C=1.0

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 📊 Visualizations

The project generates two main visualizations:

1. **Model Performance Comparison**: Bar charts and radar plots comparing all metrics
2. **Confusion Matrices**: Detailed breakdown of predictions for each model

## 🎯 Example Predictions

The system can classify various types of news:

```python
# Real news example
text1 = "Scientists discover new planet that could support life..."
result1 = detector.predict_news(text1)
# Output: Real News (95% confidence)

# Fake news example  
text2 = "BREAKING: Government conspiracy involving aliens..."
result2 = detector.predict_news(text2)
# Output: Fake News (92% confidence)
```

## 🔍 Key Features Explained

### TF-IDF Vectorization
- **Term Frequency (TF)**: How often a word appears in a document
- **Inverse Document Frequency (IDF)**: How rare a word is across all documents
- **Combination**: TF-IDF gives higher scores to words that are frequent in a document but rare across the corpus

### Model Selection
- **Logistic Regression**: Fast, interpretable, good baseline
- **Naive Bayes**: Excellent for text classification, handles high-dimensional data well
- **SVM**: Powerful for complex patterns, good generalization

## 🚨 Limitations and Considerations

1. **Domain Specificity**: Models are trained on specific datasets and may not generalize to all types of news
2. **Language Dependency**: Currently optimized for English text
3. **Context Awareness**: Models analyze text content but may miss contextual nuances
4. **Bias**: Performance depends on training data quality and representativeness

## 🔮 Future Enhancements

- **Deep Learning Models**: Implement LSTM, BERT, or transformer models
- **Multi-language Support**: Extend to other languages
- **Real-time Classification**: Web interface for live news analysis
- **Ensemble Methods**: Combine multiple models for better accuracy
- **Feature Engineering**: Add sentiment analysis, readability scores, etc.

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Fake News Detection Research](https://github.com/Heisenberg2003/Fake_News_Analysis_Using_ML)

## 👥 Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving model performance
- Enhancing visualizations
- Adding more evaluation metrics

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This tool is designed for educational and research purposes. Always verify news from multiple reliable sources before making important decisions based on automated classifications.
