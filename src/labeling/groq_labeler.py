"""
Groq API integration for C3 label generation
Fast inference with generous free tier limits
"""
from groq import Groq
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

# Configure Groq API
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Check your .env file.")

client = Groq(api_key=api_key)


class GroqLabeler:
    """Handles C3 labeling using Groq API."""
    
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        """
        Initialize Groq labeler.
        
        Args:
            model_name: Groq model to use (llama-3.3-70b-versatile by default)
        """
        self.model_name = model_name
        self.client = client
        self.retry_delay = 2  # seconds between retries
        self.max_retries = 3
        
    def label_transcript(
        self, 
        transcript: str, 
        system_prompt: str,
        user_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate C3 labels for a transcript using Groq.
        
        Args:
            transcript: The interview/speech transcript
            system_prompt: System instructions for the model
            user_prompt: User prompt template
            
        Returns:
            Dictionary with C3 scores and reasoning, or None if failed
        """
        
        for attempt in range(self.max_retries):
            try:
                # Create chat completion
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for consistent scoring
                    max_tokens=1024,
                    top_p=0.95,
                    response_format={"type": "json_object"}  # Force JSON output
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Parse JSON
                result = json.loads(response_text)
                
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
        delay: float = 0.5  # Groq is fast, shorter delay
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
