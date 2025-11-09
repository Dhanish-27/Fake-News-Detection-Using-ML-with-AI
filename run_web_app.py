import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    # Map package names to their import names
    package_mapping = {
        'flask': 'flask',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # scikit-learn is imported as sklearn
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("ERROR: Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTIP: Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_models():
    """Check if trained models exist."""
    model_files = [
        'fake_news_model_vectorizer.pkl',
        'fake_news_model_logistic_regression.pkl',
        'fake_news_model_naive_bayes.pkl',
        'fake_news_model_svm.pkl'
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print("WARNING: Trained models not found:")
        for model in missing_models:
            print(f"   - {model}")
        print("\nTIP: Train models first by running:")
        print("   python fake_news_detection.py")
        print("\nNOTE: You can still start the web app, but models need to be trained first.")
        return False
    
    return True

def check_datasets():
    """Check if dataset files exist."""
    dataset_files = [
        'NLP_Analysis/True.csv',
        'NLP_Analysis/Fake.csv'
    ]
    
    missing_datasets = []
    for dataset_file in dataset_files:
        if not os.path.exists(dataset_file):
            missing_datasets.append(dataset_file)
    
    if missing_datasets:
        print("ERROR: Dataset files not found:")
        for dataset in missing_datasets:
            print(f"   - {dataset}")
        print("\nTIP: Make sure your CSV files are in the NLP_Analysis/ folder")
        return False
    
    return True

def main():
    """Main function to start the web application."""
    print("="*60)
    print("FAKE NEWS DETECTION WEB APPLICATION")
    print("="*60)
    
    # Check requirements
    print("Checking requirements...")
    if not check_requirements():
        print("\nERROR: Requirements check failed!")
        return 1
    
    print("SUCCESS: All required packages are installed")
    
    # Check datasets
    print("\nChecking datasets...")
    datasets_ok = check_datasets()
    if not datasets_ok:
        print("\nERROR: Dataset check failed!")
        return 1
    
    print("SUCCESS: Dataset files found")
    
    # Check models
    print("\nChecking trained models...")
    models_ok = check_models()
    if models_ok:
        print("SUCCESS: All models are trained and ready")
    else:
        print("WARNING: Models not trained yet - you'll need to train them first")
    
    # Start the Flask application
    print("\nStarting Flask web application...")
    print("The app will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='127.0.0.1', port=5000)
    except ImportError as e:
        print(f"ERROR: Error importing Flask app: {e}")
        print("TIP: Make sure app.py is in the current directory")
        return 1
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"ERROR: Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
