# quick_test_openai.py - Quick test for OpenAI API
"""
Quick test to verify OpenAI API is working.
Make sure you have OPENAI_API_KEY set in your .env file.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_openai():
    """Test OpenAI API connection"""
    print("="*60)
    print("TESTING OPENAI API")
    print("="*60)
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("\n❌ OPENAI_API_KEY not found in .env file!")
        print("\nTo fix this:")
        print("1. Go to: https://platform.openai.com")
        print("2. Get your API key")
        print("3. Edit .env file and add:")
        print("   OPENAI_API_KEY=your_actual_key_here")
        return False
    
    print(f"\n✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test the API
    try:
        from openai import OpenAI
        
        print("\n🤖 Testing OpenAI API connection...")
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model='gpt-4o-mini',  # Using cheaper model for testing
            messages=[
                {'role': 'system', 'content': 'You are a fact-checker.'},
                {'role': 'user', 'content': 'Is the Earth flat? Answer in one sentence with TRUE or FALSE.'}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        print(f"\n✅ API Response: {answer}")
        print("\n🎉 SUCCESS! OpenAI API is working correctly!")
        print("\nYou can now run: python test_current_events.py")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- Network connection problem")
        print("- Insufficient credits")
        return False

if __name__ == '__main__':
    test_openai()
