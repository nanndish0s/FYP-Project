"""Simple test of Gemini API"""
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import os

# Load API key
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv('GEMINI_API_KEY')
print(f"API key loaded: {bool(api_key)}")
print(f"API key (first 10 chars): {api_key[:10] if api_key else 'None'}...")

# Configure and test
genai.configure(api_key=api_key)

# Test simple generation
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Say hello in one sentence")
print(f"\nTest response: {response.text}")
