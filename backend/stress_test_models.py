import sys
import os
import numpy as np
import pandas as pd

# Add project root to path to allow importing backend modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from backend.app import calibrate_score
except ImportError:
    print("ERROR: Could not import backend modules.")
    sys.exit(1)

def run_stress_test():
    scenarios = [
        {
            "name": "Target: Curiosity (High Quality)",
            "question_id": "q1_curiosity",
            "transcript": "I became fascinated with WebAssembly after reading about its performance potential. I wondered if it could match native speed in browsers. This curiosity drove me to spend weekends building a proof-of-concept image processing library. I experimented with Rust and benchmarked it against JavaScript.",
            "expected": "High (4.5+)"
        },
        {
            "name": "Target: Curiosity (Off-Topic High Quality)",
            "question_id": "q1_curiosity",
            "transcript": "To bake a perfect croissant, one must ensure the butter is at the precise temperature for lamination. The dough requires multi-stage fermentation to develop complex fruity esters. I spent months perfecting this delicate process, experimenting with different hydration levels and folding techniques.",
            "expected": "Penalty (< 2.0) - Irrelevant"
        },
        {
            "name": "Target: Critical Thinking (High Quality)",
            "question_id": "q2_critical_thinking",
            "transcript": "My approach to debugging involves a systematic isolation of services. I first analyze the logs to identify the point of failure. I then trace the request through the middleware to see if the root cause is a database deadlock or a network timeout. This logical flow ensures efficient resolution.",
            "expected": "High (4.5+)"
        },
        {
            "name": "Target: Critical Thinking (Nonsense)",
            "question_id": "q2_critical_thinking",
            "transcript": "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it.",
            "expected": "Penalty (1.5) - Placeholder"
        },
        {
            "name": "Target: Creativity (Repetitive Gaming)",
            "question_id": "q3_creativity",
            "transcript": "Creative solution. Creative solution. Creative solution. Creative solution. Creative solution. Creative solution. Creative solution. Creative solution. Creative solution. Creative solution. Creative solution. Creative solution!",
            "expected": "Penalty (1.5) - Repetition"
        },
        {
            "name": "Target: Creativity (Low Effort)",
            "question_id": "q3_creativity",
            "transcript": "I just kind of did it differently. It was cool.",
            "expected": "Low (~2.0) - Brief"
        }
    ]

    print("C3 MODEL STRESS TEST REPORT")
    print("=" * 80)
    print(f"{'Scenario Name':<40} | {'Raw AI*':<8} | {'Final Score':<12} | {'Result'}")
    print("-" * 80)

    for s in scenarios:
        transcript = s["transcript"]
        words = transcript.lower().split()
        word_count = len(words)
        vocab_diversity = len(set(words)) / word_count if word_count > 0 else 0
        
        # Assume a base raw model score of 3.1 (Typical conservative model output)
        raw_base = 3.1
        
        final = calibrate_score(
            raw_base, 
            word_count, 
            vocab_diversity, 
            transcript=transcript, 
            question_id=s["question_id"]
        )
        
        status = "✅ PASS"
        if "High" in s["expected"] and final < 4.0: status = "❌ FAIL"
        if "Penalty" in s["expected"] and final > 2.5: status = "❌ FAIL"
        
        print(f"{s['name']:<40} | {raw_base:<8} | {final:<12.2f} | {status}")

    print("-" * 80)
    print("*Raw AI assumes a neutral model prediction of 3.1 for all cases.")
    print("This demonstrates how the Calibration Layer handles different content qualities.")

if __name__ == "__main__":
    run_stress_test()
