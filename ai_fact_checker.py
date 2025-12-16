# ai_fact_checker.py - AI-powered fact-checking module
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
import config

class AIFactChecker:
    """
    AI-powered fact-checking system with multi-provider support
    Supports: Grok (xAI), GPT-4 (OpenAI), Gemini (Google)
    """
    
    def __init__(self, provider: str = None):
        """
        Initialize AI fact-checker with specified provider
        
        Args:
            provider: 'grok', 'openai', or 'gemini'. If None, uses first available.
        """
        self.provider = provider or config.DEFAULT_AI_PROVIDER
        self.providers = config.get_available_providers()
        
        if not self.providers:
            print("⚠️  Warning: No AI providers configured!")
            print("   Set API keys in .env file to enable AI fact-checking")
            self.enabled = False
            return
        
        # Use fallback if requested provider not available
        if self.provider not in self.providers:
            self.provider = self.providers[0]
            print(f"ℹ️  Provider '{provider}' not available, using '{self.provider}'")
        
        self.config = config.get_provider_config(self.provider)
        self.enabled = True
        
        print(f"✅ AI Fact-Checker initialized with {self.provider.upper()}")
    
    def _call_grok(self, prompt: str) -> Optional[str]:
        """Call Grok API (xAI)"""
        try:
            headers = {
                'Authorization': f'Bearer {self.config["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': self.config['model'],
                'messages': [
                    {'role': 'system', 'content': 'You are an expert fact-checker. Respond in valid JSON format only.'},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': self.config['max_tokens'],
                'temperature': self.config['temperature']
            }
            
            response = requests.post(
                self.config['endpoint'],
                headers=headers,
                json=payload,
                timeout=config.HYBRID_CONFIG['ai_timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"❌ Grok API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Grok API exception: {str(e)}")
            return None
    
    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI GPT-4 API"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.config['api_key'])
            
            response = client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {'role': 'system', 'content': 'You are an expert fact-checker. Respond in valid JSON format only.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ OpenAI API exception: {str(e)}")
            return None
    
    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Google Gemini API"""
        try:
            from google import genai
            from google.genai import types
            
            # Initialize client
            client = genai.Client(api_key=self.config['api_key'])
            
            # Generate response
            response = client.models.generate_content(
                model=self.config['model'],
                contents=f"You are an expert fact-checker. Respond in valid JSON format only.\n\n{prompt}",
                config=types.GenerateContentConfig(
                    temperature=self.config['temperature'],
                    max_output_tokens=self.config['max_tokens']
                )
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini API exception: {str(e)}")
            return None
    
    def _call_ai(self, prompt: str) -> Optional[str]:
        """Call the configured AI provider"""
        if not self.enabled:
            return None
        
        if self.provider == 'grok':
            return self._call_grok(prompt)
        elif self.provider == 'openai':
            return self._call_openai(prompt)
        elif self.provider == 'gemini':
            return self._call_gemini(prompt)
        else:
            print(f"❌ Unknown provider: {self.provider}")
            return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from AI response, handling markdown code blocks"""
        if not response:
            return None
        
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            try:
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response
                
                return json.loads(json_str)
            except Exception as e:
                print(f"❌ Failed to parse JSON response: {str(e)[:100]}")
                return None
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract verifiable factual claims from text"""
        prompt = f"""Extract all verifiable factual claims from this text:

\"\"\"{text}\"\"\"

Return a JSON object with:
{{
    "claims": ["claim 1", "claim 2", ...]
}}

Only include statements that can be fact-checked, not opinions or subjective statements."""
        
        response = self._call_ai(prompt)
        parsed = self._parse_json_response(response)
        
        if parsed and 'claims' in parsed:
            return parsed['claims']
        return []
    
    def verify_claim(self, claim: str) -> Dict:
        """Verify a single claim"""
        prompt = f"""Verify this factual claim:

\"{claim}\"

Return JSON:
{{
    "claim": "{claim}",
    "verdict": "TRUE" | "FALSE" | "UNVERIFIED" | "MISLEADING",
    "confidence": 0-100,
    "reasoning": "brief explanation"
}}"""
        
        response = self._call_ai(prompt)
        parsed = self._parse_json_response(response)
        
        if parsed:
            return parsed
        
        return {
            'claim': claim,
            'verdict': 'UNVERIFIED',
            'confidence': 0,
            'reasoning': 'AI verification failed'
        }
    
    def analyze_news(self, text: str) -> Dict:
        """
        Complete AI analysis of news text
        
        Returns:
            Dictionary with prediction, confidence, claims, verification, explanation
        """
        if not self.enabled:
            return {
                'success': False,
                'error': 'AI fact-checking not enabled. Set API keys in .env file.',
                'prediction': 'UNKNOWN',
                'confidence': 0
            }
        
        # Use the comprehensive fact-checking prompt
        prompt = config.FACT_CHECK_PROMPT.format(text=text[:2000])  # Limit length
        
        print(f"🤖 Analyzing with {self.provider.upper()}...")
        start_time = time.time()
        
        response = self._call_ai(prompt)
        elapsed = time.time() - start_time
        
        if not response:
            return {
                'success': False,
                'error': f'{self.provider} API call failed',
                'prediction': 'UNKNOWN',
                'confidence': 0
            }
        
        parsed = self._parse_json_response(response)
        
        if not parsed:
            return {
                'success': False,
                'error': 'Failed to parse AI response',
                'prediction': 'UNKNOWN',
                'confidence': 0,
                'raw_response': response[:500]
            }
        
        # Extract overall assessment
        overall = parsed.get('overall_assessment', {})
        
        result = {
            'success': True,
            'provider': self.provider,
            'prediction': overall.get('prediction', 'UNKNOWN'),
            'confidence': overall.get('confidence', 50),
            'explanation': overall.get('explanation', 'No explanation provided'),
            'claims': parsed.get('claims', []),
            'verification': parsed.get('verification', []),
            'source_credibility': parsed.get('source_credibility', {}),
            'sources': parsed.get('sources', []),
            'response_time': round(elapsed, 2)
        }
        
        print(f"✅ AI Analysis complete in {elapsed:.2f}s")
        return result
    
    def quick_check(self, text: str) -> Tuple[str, float]:
        """
        Quick fact-check returning just prediction and confidence
        
        Returns:
            (prediction, confidence) tuple
        """
        result = self.analyze_news(text)
        
        if result['success']:
            return result['prediction'], result['confidence']
        else:
            return 'UNKNOWN', 0.0


def test_ai_fact_checker():
    """Test the AI fact-checker with sample texts"""
    print("="*60)
    print("TESTING AI FACT-CHECKER")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            'text': 'BREAKING: Aliens confirmed in Area 51 by TOP SECRET government documents!',
            'expected': 'FAKE'
        },
        {
            'text': 'Scientists at Harvard University discovered a new species of amphibian in the Amazon rainforest.',
            'expected': 'TRUE'
        },
        {
            'text': 'xAI releases Grok 5 with 6 trillion parameters in Q1 2026.',
            'expected': 'TRUE'
        }
    ]
    
    # Try each available provider
    providers = config.get_available_providers()
    
    if not providers:
        print("\n❌ No AI providers configured!")
        print("   Please set API keys in .env file")
        return
    
    for provider in providers[:1]:  # Test first available provider
        print(f"\n{'='*60}")
        print(f"Testing with {provider.upper()}")
        print(f"{'='*60}")
        
        checker = AIFactChecker(provider=provider)
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Text: {test['text'][:80]}...")
            print(f"Expected: {test['expected']}")
            
            result = checker.analyze_news(test['text'])
            
            if result['success']:
                print(f"Prediction: {result['prediction']} ({result['confidence']}% confidence)")
                print(f"Explanation: {result['explanation'][:150]}...")
                print(f"Response time: {result['response_time']}s")
                
                # Check if prediction matches expected
                match = result['prediction'] == test['expected']
                print(f"Result: {'✅ CORRECT' if match else '❌ INCORRECT'}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)
    print("AI FACT-CHECKER TEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    test_ai_fact_checker()
