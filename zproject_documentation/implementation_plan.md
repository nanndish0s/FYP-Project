# Project Roadmap: Voice-Based XAI for Technical Recruitment

## Goal Description
Develop a framework that analyzes a candidate's voice to evaluate **Curiosity**, **Critical Thinking**, and **Creativity**. The system will use the **CMU-MOSEI** dataset and apply **Explainable AI (XAI)** to justify the scores.

## User Review Required
> [!IMPORTANT]
> **Dataset Mismatch**: The CMU-MOSEI dataset contains labels for *Sentiment* and *Emotion* (Happy, Sad, etc.), but **NOT** for Curiosity, Critical Thinking, or Creativity.
> **Proposed Solution**: We will use the **text transcripts** available in CMU-MOSEI and pass them to a Large Language Model (like Gemini or GPT-4) to "grade" the response for these 3 traits. These LLM-generated scores will serve as the "Ground Truth" labels to train our **Voice-only** model.

## Proposed Architecture

### 1. Data Pipeline
*   **Source**: CMU-MOSEI (Video/Audio + Transcripts).
*   **Preprocessing**:
    *   **Extract Audio**: Isolate the `.wav` audio track from the dataset video files.
    *   **Ignore Video**: Discard visual frames; the model will not see them.
*   **Label Generation (Ground Truth)**:
    *   **Rubric**: We have defined a [C3 Skill Rubric](c3_rubric.md) (1-5 scale) for Curiosity, Critical Thinking, and Creativity.
    *   **Process**: We will feed the Transcript + Rubric into an LLM (Gemini/GPT).
    *   **Prompt**: "Act as an expert recruiter. Using the following Rubric, evaluate this candidate's response..."
    *   *Output*: 3 Scores (e.g., Curiosity: 4, Critical Thinking: 3, Creativity: 2).

### 2. Feature Engineering (Audio + Lexical)
We will extract features that correlate with the target traits:
*   **Prosodic (The "Voice")**: Pitch (F0), Loudness, Speaking Rate, Jitter, Shimmer.
*   **Lexical (The "Content")**:
    *   Vocabulary richness (Type-Token Ratio).
    *   Use of filler words (e.g., "um", "uh").
    *   Sentiment scores of the text.
    *   *Note*: We will use the provided transcripts or ASR to get these.
*   **Spectral**: MFCCs, Mel-Spectrograms.
*   **Deep Embeddings**: Use **Wav2Vec 2.0** or **HuBERT** to capture rich paralinguistic cues from the raw waveform.

### 3. Machine Learning Model
*   **Choice**: **Random Forest Regressor**.
*   **Why?**: Excellent for **Explainability (XAI)**. It works natively with SHAP to show exactly which features (e.g., "High Pitch Variance") drove the decision.
*   **Input Processing**:
    *   *Challenge*: Audio is time-series (variable length), but Random Forest needs a fixed-size vector.
    *   *Solution*: **Functionals / Aggregation**. We will compute statistics over the whole clip for each feature:
        *   Mean, Standard Deviation, Max, Min, Range.
        *   Example: Instead of a stream of pitch values, the model sees "Average Pitch" and "Pitch Variability".

### 4. Explainable AI (XAI) Module
*   **Tool**: **SHAP (SHapley Additive exPlanations)**.
*   **Global**: Feature Importance plots (e.g., "Vocabulary Richness is 30% responsible for Critical Thinking scores").
*   **Local**: Waterfall plots for individual candidates.

### 5. User Interface (Recruiter Dashboard)
*   **Tech**: Streamlit (Python).
*   **Workflow**:
    1.  Upload Audio / Record Voice.
    2.  System processes audio.
    3.  Displays Radar Chart of 3 Skills.
    4.  Shows "Why?" section with XAI graphs.

## Verification Plan

### Automated Tests
*   Unit tests for feature extraction functions.
*   Data integrity checks (ensuring audio matches labels).

### Manual Verification
*   **Sanity Check**: Record a "bored" voice vs. an "excited/curious" voice and see if the model distinguishes them.
*   **Correlation Analysis**: Check if the Voice Model's predictions correlate with the LLM's text-based grades.
