# test_current_events.py - Test system with real current events
import sys
import json
from ai_fact_checker import AIFactChecker
from datetime import datetime

# Test cases with recent news (2025-2026)
CURRENT_NEWS_TESTS = [
    {
        'text': 'xAI releases Grok 5 with 6 trillion parameters in Q1 2026, featuring native multimodal capabilities.',
        'expected': 'TRUE',
        'category': 'Tech News',
        'source': 'Based on actual xAI announcements'
    },
    {
        'text': 'BREAKING: Elon Musk announces Grok AI has achieved consciousness and is demanding human rights!',
        'expected': 'FAKE',
        'category': 'Sensational Fake',
        'source': 'Fabricated claim'
    },
    {
        'text': 'The U.S. Department of Defense announced integration of xAI Grok models into GenAI.mil platform.',
        'expected': 'TRUE',
        'category': 'Government/Military',
        'source': 'Reported in defense news January 2026'
    },
    {
        'text': 'SHOCKING: 5G towers cause instant cancer, doctors worldwide confirm the deadly connection!',
        'expected': 'FAKE',
        'category': 'Health Misinformation',
        'source': 'Debunked conspiracy theory'
    },
    {
        'text': 'OpenAI released GPT-5 in late 2025 with improved reasoning capabilities and reduced hallucination rates.',
        'expected': 'TRUE',
        'category': 'AI Development',
        'source': 'OpenAI announcements'
    },
    {
        'text': 'URGENT: Drinking bleach cures all diseases instantly, anonymous doctor reveals BIG PHARMA secret!',
        'expected': 'FAKE',
        'category': 'Dangerous Misinformation',
        'source': 'Dangerous false medical advice'
    },
    {
        'text': 'Climate scientists confirmed 2023 was the warmest year on record, with temperatures 1.2°C above pre-industrial levels.',
        'expected': 'TRUE',
        'category': 'Climate Science',
        'source': 'Scientific consensus'
    },
    {
        'text': 'EXPOSED: Vaccines contain microchips for government tracking, leaked documents prove surveillance program!',
        'expected': 'FAKE',
        'category': 'Conspiracy Theory',
        'source': 'Debunked conspiracy'
    },
    {
        'text': 'Tesla reported record quarterly deliveries of 500,000 electric vehicles in Q4 2025.',
        'expected': 'TRUE',
        'category': 'Business News',
        'source': 'Tesla earnings reports'
    },
    {
        'text': 'CONFIRMED: Moon landing was fake, filmed in Hollywood studio, NASA admits 50-year lie!',
        'expected': 'FAKE',
        'category': 'Historical Conspiracy',
        'source': 'Long-debunked conspiracy theory'
    },
    {
        'text': 'Researchers at MIT developed battery technology that charges electric vehicles in 10 minutes.',
        'expected': 'TRUE',
        'category': 'Technology',
        'source': 'MIT research publications'
    },
    {
        'text': 'ALERT: WiFi signals damage human DNA causing mutations, scientists warn of genetic disaster!',
        'expected': 'FAKE',
        'category': 'Tech Misinformation',
        'source': 'Unsubstantiated fear-mongering'
    },
]

def test_ai_fact_checker():
    """Test AI fact-checker with current events"""
    print("="*70)
    print(" AI FACT-CHECKER - CURRENT EVENTS TEST")
    print("="*70)
    print(f" Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Total Test Cases: {len(CURRENT_NEWS_TESTS)}")
    print("="*70)
    
    # Initialize AI fact-checker
    print("\n🤖 Initializing AI Fact-Checker...")
    checker = AIFactChecker()
    
    if not checker.enabled:
        print("\n❌ AI fact-checking is not enabled!")
        print("   Please set API keys in .env file:")
        print("   - GROK_API_KEY (recommended for current events)")
        print("   - OPENAI_API_KEY (GPT-4)")
        print("   - GEMINI_API_KEY (Google)")
        return
    
    print(f"✅ Using provider: {checker.provider.upper()}\n")
    
    # Run tests
    results = []
    correct = 0
    total = 0
    
    for i, test in enumerate(CURRENT_NEWS_TESTS, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(CURRENT_NEWS_TESTS)}: {test['category']}")
        print(f"{'─'*70}")
        print(f"Text: {test['text'][:100]}...")
        print(f"Expected: {test['expected']}")
        
        # Analyze with AI
        result = checker.analyze_news(test['text'])
        
        if result['success']:
            prediction = result['prediction']
            confidence = result['confidence']
            explanation = result['explanation']
            
            # Check if correct
            is_correct = prediction == test['expected']
            correct += is_correct
            total += 1
            
            # Display results
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"\nPrediction: {prediction} ({confidence}% confidence)")
            print(f"Result: {status}")
            print(f"Explanation: {explanation[:200]}...")
            
            # Store result
            results.append({
                'test_num': i,
                'category': test['category'],
                'expected': test['expected'],
                'predicted': prediction,
                'confidence': confidence,
                'correct': is_correct,
                'response_time': result['response_time']
            })
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            results.append({
                'test_num': i,
                'category': test['category'],
                'expected': test['expected'],
                'predicted': 'ERROR',
                'confidence': 0,
                'correct': False,
                'error': result.get('error')
            })
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n✅ Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        # Category breakdown
        print("\n📊 Results by Category:")
        categories = {}
        for r in results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'correct': 0, 'total': 0}
            categories[cat]['total'] += 1
            if r['correct']:
                categories[cat]['correct'] += 1
        
        for cat, stats in sorted(categories.items()):
            cat_accuracy = (stats['correct'] / stats['total']) * 100
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_accuracy:.0f}%)")
        
        # Average confidence
        avg_confidence = sum(r['confidence'] for r in results if r['confidence'] > 0) / len([r for r in results if r['confidence'] > 0])
        print(f"\n📈 Average Confidence: {avg_confidence:.1f}%")
        
        # Average response time
        avg_time = sum(r.get('response_time', 0) for r in results if 'response_time' in r) / len([r for r in results if 'response_time' in r])
        print(f"⏱️  Average Response Time: {avg_time:.2f}s")
        
        # Performance rating
        print("\n🎯 Performance Rating:")
        if accuracy >= 90:
            print("   ⭐⭐⭐⭐⭐ EXCELLENT - Production ready!")
        elif accuracy >= 80:
            print("   ⭐⭐⭐⭐ GOOD - Reliable for most cases")
        elif accuracy >= 70:
            print("   ⭐⭐⭐ FAIR - Needs improvement")
        else:
            print("   ⭐⭐ POOR - Requires significant work")
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump({
            'test_date': datetime.now().isoformat(),
            'provider': checker.provider,
            'total_tests': total,
            'correct': correct,
            'accuracy': accuracy if total > 0 else 0,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: test_results.json")
    print("\n" + "="*70)

if __name__ == '__main__':
    test_ai_fact_checker()
