# Voice-Based XAI Framework for AI Recruitment (Software Engineering)

## Overview
This project is an **AI-powered recruitment system** that evaluates **software engineering candidates'** soft skills (Curiosity, Critical Thinking, and Creativity) based on multimodal voice analysis.

## Architecture
- **Frontend**: Modern React Application (TypeScript + Tailwind CSS)
- **Backend**: Python Flask API
- **AI Engine**: Random Forest Regressors with Calibrated Inference Layer

## Project Structure
*   `frontend/`: React application code
*   `backend/`: Flask API and AI model inference logic
*   `data/`: Processed datasets and metadata
*   `models/`: Trained Random Forest models
*   `src/`: Core feature extraction and engineering scripts

## Getting Started

### 1. Start the Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

## Midpoint Validation
To verify the system's "judgment" engine across multiple scenarios, you can run the stress test:
```bash
python backend/stress_test_models.py
```
