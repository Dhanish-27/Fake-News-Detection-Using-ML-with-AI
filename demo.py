"""
Demo script for Fake News Detection System
==========================================

This script demonstrates how to use the FakeNewsDetector class
for quick testing and demonstration purposes.
"""

from fake_news_detection import FakeNewsDetector

def demo_predictions():
    """Demonstrate the fake news detection system with sample texts."""
    
    print("="*80)
    print("FAKE NEWS DETECTION DEMO")
    print("="*80)
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    try:
        # Load and prepare data
        print("Loading data and training models...")
        detector.load_data()
        detector.prepare_data()
        detector.train_models()
        
        print("\nModels trained successfully!")
        
        # Sample texts for demonstration
        sample_news = [
            {
                "title": "Real News Example",
                "text": "Scientists at NASA have successfully launched a new Mars rover mission. The rover, named Perseverance, will search for signs of ancient life on the red planet and collect rock samples for future return to Earth. The mission represents a major milestone in space exploration and will help scientists better understand Mars' geological history.",
                "expected": "Real News"
            },
            {
                "title": "Fake News Example", 
                "text": "BREAKING: Local man discovers that the government has been hiding aliens in underground bases for decades! Sources say that reptilian overlords are controlling world leaders and the media is covering it up. This shocking revelation explains everything from climate change to the stock market fluctuations. The truth is finally coming out!",
                "expected": "Fake News"
            },
            {
                "title": "Technology News",
                "text": "Apple Inc. reported record quarterly earnings today, driven by strong iPhone sales and growing services revenue. The company's stock price rose 3% in after-hours trading following the announcement. CEO Tim Cook highlighted the company's commitment to sustainability and privacy in his earnings call.",
                "expected": "Real News"
            },
            {
                "title": "Conspiracy Theory",
                "text": "URGENT: The mainstream media doesn't want you to know this, but a secret cabal of billionaires is using 5G towers to control our minds! The proof is everywhere - just look at how people are glued to their phones. Wake up, sheeple! The deep state is behind this global mind control experiment.",
                "expected": "Fake News"
            }
        ]
        
        print("\n" + "="*60)
        print("TESTING SAMPLE NEWS ARTICLES")
        print("="*60)
        
        # Test each sample with all models
        for i, news in enumerate(sample_news, 1):
            print(f"\n{i}. {news['title']}")
            print("-" * 50)
            print(f"Expected: {news['expected']}")
            print(f"Text: {news['text'][:100]}...")
            print()
            
            # Test with all three models
            for model_name in ['Logistic Regression', 'Naive Bayes', 'SVM']:
                try:
                    result = detector.predict_news(news['text'], model_name)
                    print(f"{model_name:20} | {result['prediction']:10} | "
                          f"Confidence: {result['confidence']['Real News']:.1%} Real")
                except Exception as e:
                    print(f"{model_name:20} | Error: {str(e)}")
            
            print()
        
        # Show overall model performance
        print("="*60)
        print("OVERALL MODEL PERFORMANCE")
        print("="*60)
        detector.evaluate_models()
        
        # Interactive prediction
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION")
        print("="*60)
        print("Enter your own news text to test (or 'quit' to exit):")
        
        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                try:
                    result = detector.predict_news(user_input, 'Logistic Regression')
                    print(f"\nPrediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']['Real News']:.1%} Real, "
                          f"{result['confidence']['Fake News']:.1%} Fake")
                    print(f"Model Used: {result['model_used']}")
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        print("\nThank you for using the Fake News Detection Demo!")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. Please ensure True.csv and Fake.csv are in the NLP_Analysis/ folder.")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check that all dependencies are installed and datasets are available.")

if __name__ == "__main__":
    demo_predictions()
