"""
Quick test script to verify the fake news detection models work correctly.
"""

from fake_news_detection import FakeNewsDetector
import time

def quick_test():
    """Run a quick test of the fake news detection system."""
    
    print("="*60)
    print("QUICK TEST - FAKE NEWS DETECTION")
    print("="*60)
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    try:
        print("Loading data...")
        start_time = time.time()
        
        # Load and prepare data
        detector.load_data()
        detector.prepare_data()
        
        print(f"Data loaded in {time.time() - start_time:.2f} seconds")
        
        print("\nTraining models...")
        train_start = time.time()
        
        # Train models
        detector.train_models()
        
        print(f"Models trained in {time.time() - train_start:.2f} seconds")
        
        # Quick evaluation
        print("\nModel Performance Summary:")
        print("-" * 40)
        for name, results in detector.results.items():
            print(f"{name:20} | Accuracy: {results['accuracy']:.3f} | F1: {results['f1_score']:.3f}")
        
        # Test predictions
        print("\nTesting predictions:")
        print("-" * 40)
        
        test_texts = [
            "Scientists discover new planet with potential for life. The discovery was made using advanced telescopes and represents a major breakthrough in astronomy.",
            "BREAKING: Government hiding aliens in secret bases! Sources say reptilian overlords control everything. Wake up people!"
        ]
        
        for i, text in enumerate(test_texts, 1):
            result = detector.predict_news(text, 'Logistic Regression')
            print(f"\nTest {i}:")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']['Real News']:.1%} Real")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
