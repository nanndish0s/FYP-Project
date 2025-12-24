"""
Debug Gemini API - check available models and permissions
"""
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

# Configure
genai.configure(api_key=api_key)

try:
    print("\nListing available models...")
    models = list(genai.list_models())
    
    print(f"\nTotal models: {len(models)}")
    
    # Filter models that support generateContent
    content_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
    
    print(f"\nModels supporting content generation: {len(content_models)}")
    print("\nAvailable models:")
    for model in content_models[:10]:  # Show first 10
        print(f"  - {model.name}")
        print(f"    Display name: {model.display_name}")
        print(f"    Description: {model.description[:80]}..." if len(model.description) > 80 else f"    Description: {model.description}")
        print()
    
except Exception as e:
    print(f"\nError listing models: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Try simple generation with first available model if any found
if 'content_models' in locals() and len(content_models) > 0:
    print("\n" + "="*70)
    print(f"Testing with first available model: {content_models[0].name}")
    print("="*70)
    
    try:
        test_model = genai.GenerativeModel(content_models[0].name)
        response = test_model.generate_content("Say hello in one sentence")
        print(f"✓ Success! Response: {response.text}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
