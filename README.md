# Voice-Based XAI Framework for AI Recruitment (Software Engineering)

## Overview
This project is an **AI-powered recruitment system** that evaluates **software engineering candidates'** soft skills—specifically **Curiosity**, **Critical Thinking**, and **Creativity (C3)**—based on voice analysis during interviews. It uses **Explainable AI (XAI)** to provide transparent justifications for the scores.

**Domain**: AI Recruitment for Technical Roles (Software Engineering)  
**Key Innovation**: Voice-based soft skills assessment with explainable predictions

## Methodology
1.  **Dataset**: CMU-MOSEI (Audio + Transcripts) - general interview/speech data for MVP
2.  **Ground Truth**: Generated using LLMs (Gemini) based on the [C3 Skill Rubric](docs/c3_rubric.md) tailored for technical candidates
3.  **Model**: Random Forest Regressor trained on aggregated Acoustic and Lexical features
4.  **XAI**: SHAP (SHapley Additive exPlanations) to explain predictions

## Project Structure
*   `data/`: Raw and processed datasets
*   `src/`: Source code for feature extraction and modeling
    *   `features/`: Scripts for extracting Prosodic and Lexical features
    *   `models/`: Training and evaluation scripts
    *   `labeling/`: LLM-based C3 label generation
*   `notebooks/`: Jupyter notebooks for exploration

## Installation
```bash
pip install -r requirements.txt
```
