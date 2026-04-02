# Tasks: Voice-Based XAI for Recruitment

- [ ] **Phase 1: Project Setup & Data Preparation**
    - [x] Initialize project structure (Python, Git)
    - [ ] Download/Access CMU-MOSEI dataset sample
    - [ ] **CRITICAL**: Solve Label Gap (CMU-MOSEI lacks Curiosity/Critical Thinking/Creativity labels)
        - [ ] Plan: Use LLM (Gemini/GPT) to analyze transcripts and generate pseudo-labels for the 3 traits
    - [ ] **Audio Extraction**: Convert CMU-MOSEI clips to `.wav` format (discarding video).

- [ ] **Phase 2: Feature Extraction (Audio + Lexical)**
    - [ ] **Acoustic**: Extract Pitch, Energy, MFCCs (Librosa/OpenSMILE).
    - [ ] **Lexical**: Extract word counts, filler words, vocabulary complexity (using NLTK/Spacy on transcripts).
    - [ ] Explore Deep Embeddings (Wav2Vec2 or HuBERT).

- [ ] **Phase 3: Model Development (Random Forest)**
    *   [ ] **Data Prep**: Aggregate time-series features into fixed vectors (Mean/Std/Max).
    *   [ ] **Training**: Train Random Forest Regressor (Input: Aggregated Audio+Lexical -> Output: 3 Scores).
    *   [ ] **Tuning**: Grid Search for hyperparameters (Trees, Depth).

# Task: Project Documentation
- [x] Draft Chapter 5: Implementation / Experimental Setup
    - [x] Document Technology Selection & Justification
    - [x] Detail Data Selection (RecruitView integration)
    - [x] Explain Core Modules (37-feature extraction, Calibration)
    - [x] Describe React Frontend & User Experience
- [x] Initial Exploration
    - [x] Read README.md
    - [x] Explore backend structure
    - [x] Explore frontend structure
- [x] Deep Dive into Core Logic
    - [x] Analyze entry points (backend/app.py)
    - [x] Analyze feature extraction scripts
    - [x] Understand model training and inference
- [x] Refine Dataset Metadata Mapping
    - [x] Decouple Curiosity and Creativity in `ingest_recruitview.py`
    - [x] Retrain models for finalized scoring baseline
- [x] Review Deployment Context (from previous conversations)
- [x] Integrate RecruitView models (2,011 samples) and 37-feature pipeline
    - [x] Update `backend/app.py` for 37-feature extraction and model loading
    - [x] Align `src/features/extract_live_features.py` with RecruitView schema
    - [x] Update Streamlit frontend (`app.py`, `src/app_live_tab.py`) for production models
    - [x] Handle missing metadata/reasoning in `recruitview_metadata.csv`
- [x] Verify End-to-End Prediction Flow
- [x] Debug Live Assessment Score Discrepancy
    - [x] Compare live feature ranges with training data distribution
    - [x] Audit vocal vs lexical feature weights in production models
    - [x] Refine LiveFeatureExtractor normalization if needed
- [x] Final Presentation Prep (Walkthrough update)
- [x] Final Summary

- [ ] **Phase 4: Explainable AI (XAI) Integration**
    - [ ] Implement **TreeExplainer** (SHAP optimized for Random Forests).
    - [ ] Generate Feature Importance & Waterfall plots.

- [ ] **Phase 5: Evaluation & Reporting**
    - [ ] Calculate Metrics (MAE, MSE, Accuracy, F1)
    - [ ] Compare Model predictions vs Pseudo-labels

- [ ] **Phase 6: Demonstration Interface**
    - [x] Build Streamlit/Gradio Web App
    - [x] Real-time voice recording and analysis demo
    - [x] **Calibration**: Adjust scores for demo responsiveness

# Task: Mid-Point Submission Refinement
- [x] Systematic Removal of XAI Features
- [x] Score Calibration for Live Demo
    - [x] Implement initial calibration function
    - [x] Add heuristic Lexical Boost
    - [x] Refine penalties for clearer "Low" tier differentiation
- [x] Demo Script Preparation (High/Medium/Low)
- [x] System Evaluation & Visual Graph Generation (Chapter 7)
