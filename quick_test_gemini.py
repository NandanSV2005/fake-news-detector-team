# quick_test_gemini.py - Quick test for Gemini API
"""
Quick test to verify Gemini API is working.
Make sure you have GEMINI_API_KEY set in your .env file.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini():
    """Test Gemini API connection"""
    print("="*60)
    print("TESTING GEMINI API")
    print("="*60)
    
    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found in .env file!")
        print("\nTo fix this:")
        print("1. Go to: https://ai.google.dev")
        print("2. Click 'Get API Key'")
        print("3. Copy your API key")
        print("4. Edit .env file and add:")
        print("   GEMINI_API_KEY=your_actual_key_here")
        return False
    
    print(f"\n✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test the API
    try:
        from google import genai
        from google.genai import types
        
        print("\n🤖 Testing Gemini API connection...")
        
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents="Is the Earth flat? Answer in one sentence with TRUE or FALSE.",
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=100
            )
        )
        
        print(f"\n✅ API Response: {response.text}")
        print("\n🎉 SUCCESS! Gemini API is working correctly!")
        print("\nYou can now run: python test_current_events.py")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- Network connection problem")
        print("- API quota exceeded")
        return False

if __name__ == '__main__':
    test_gemini()
