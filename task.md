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

- [x] Initial Exploration
    - [x] Read README.md
    - [x] Explore backend structure
    - [x] Explore frontend structure
- [x] Deep Dive into Core Logic
    - [x] Analyze entry points (backend/app.py)
    - [x] Analyze feature extraction scripts
    - [x] Understand model training and inference
- [x] Review Deployment Context (from previous conversations)
- [x] Final Summary

- [ ] **Phase 4: Explainable AI (XAI) Integration**
    - [ ] Implement **TreeExplainer** (SHAP optimized for Random Forests).
    - [ ] Generate Feature Importance & Waterfall plots.

- [ ] **Phase 5: Evaluation & Reporting**
    - [ ] Calculate Metrics (MAE, MSE, Accuracy, F1)
    - [ ] Compare Model predictions vs Pseudo-labels

- [ ] **Phase 6: Demonstration Interface**
    - [ ] Build Streamlit/Gradio Web App
    - [ ] Real-time voice recording and analysis demo
