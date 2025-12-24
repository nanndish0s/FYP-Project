"""
Gemini API integration for C3 label generation
"""
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Check your .env file.")
genai.configure(api_key=api_key)


class GeminiLabeler:
    """Handles C3 labeling using Gemini API."""
    
    def __init__(self, model_name="models/gemini-2.0-flash-exp"):
        """
        Initialize Gemini labeler.
        
        Args:
            model_name: Gemini model to use (models/gemini-2.0-flash-exp by default)
        """
        self.model = genai.GenerativeModel(model_name)
        self.retry_delay = 2  # seconds between retries
        self.max_retries = 3
        
    def label_transcript(
        self, 
        transcript: str, 
        system_prompt: str,
        user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate C3 labels for a transcript using Gemini.
        
        Args:
            transcript: The interview/speech transcript
            system_prompt: System instructions for the model
            user_prompt: User prompt template
            
        Returns:
            Dictionary with C3 scores and reasoning, or None if failed
        """
        # Combine system prompt and user prompt
        # Note: Gemini doesn't have explicit system/user roles like ChatGPT
        # So we prepend system instructions to the prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        for attempt in range(self.max_retries):
            try:
                # Generate response
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': 0.3,  # Lower temperature for more consistent scoring
                        'top_p': 0.95,
                        'top_k': 40,
                        'max_output_tokens': 1024,
                    }
                )
                
                # Extract text
                response_text = response.text
                
                # Try to parse JSON
                # Look for JSON block in markdown code fence or raw
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text.strip()
                
                # Parse JSON
                result = json.loads(json_text)
                
                # Validate structure
                required_keys = ['curiosity', 'critical_thinking', 'creativity']
                if all(key in result for key in required_keys):
                    return result
                else:
                    print(f"Warning: Response missing required keys. Got: {result.keys()}")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"JSON parse error (attempt {attempt + 1}/{self.max_retries}): {e}")
                print(f"Response was: {response_text[:200]}...")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
                    
            except Exception as e:
                print(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
        
        return None
    
    def batch_label(
        self,
        transcripts: list,
        system_prompt: str,
        user_prompt_template: str,
        delay: float = 1.0
    ) -> list:
        """
        Label multiple transcripts with rate limiting.
        
        Args:
            transcripts: List of transcript texts
            system_prompt: System instructions
            user_prompt_template: Template with {transcript} placeholder
            delay: Seconds to wait between API calls
            
        Returns:
            List of results (dictionaries or None for failures)
        """
        results = []
        
        for i, transcript in enumerate(transcripts):
            print(f"Processing transcript {i+1}/{len(transcripts)}...")
            
            # Format user prompt
            user_prompt = user_prompt_template.format(transcript=transcript)
            
            # Get labels
            result = self.label_transcript(transcript, system_prompt, user_prompt)
            results.append(result)
            
            # Rate limiting
            if i < len(transcripts) - 1:
                time.sleep(delay)
        
        return results
