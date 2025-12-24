"""Test Groq API and list available models"""
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import os

# Load API key
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.getenv('GROQ_API_KEY')
print(f"API key loaded: {bool(api_key)}")

# Create client
client = Groq(api_key=api_key)

# Test simple completion
try:
    print("\nTesting with llama-3.3-70b-versatile...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello in one sentence"}],
        max_tokens=50
    )
    print(f"Success! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error with llama-3.3-70b-versatile: {e}")

# Try other models
print("\nTrying other models...")
models_to_try = [
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "llama3-70b-8192",
    "gemma2-9b-it"
]

for model in models_to_try:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print(f"✓ {model} - WORKS")
        break
    except Exception as e:
        print(f"✗ {model} - Error: {str(e)[:50]}")
