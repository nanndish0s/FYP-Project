# Complete Setup Guide - React + Flask Application

## 🚀 Quick Start

### Terminal 1: Backend (Flask API)
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Backend runs on: http://localhost:5000

### Terminal 2: Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on: http://localhost:5173

## 📁 Project Structure

```
Final Year Project Mid-Point/
├── backend/                    # Flask REST API
│   ├── app.py                  # Main API server
│   └── requirements.txt
│
├── frontend/                   # React application
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── pages/              # Page components
│   │   ├── services/           # API client
│   │   ├── App.tsx             # Main app
│   │   └── main.tsx            # Entry point
│   ├── package.json
│   └── tailwind.config.js
│
├── app.py                      # Streamlit app (legacy)
├── models/                     # ML models
└── data/                       # Dataset
```

## ✨ Features Implemented

### React Frontend
- ✅ Professional recruitment UI
- ✅ Live audio recording in browser
- ✅ C3 gauge chart visualizations
- ✅ Candidate gallery with 44 candidates
- ✅ Responsive design with Tailwind CSS
- ✅ Modern animations and transitions

### Flask Backend
- ✅ REST API with CORS support
- ✅ `/api/candidates` - List all candidates
- ✅ `/api/candidate/<id>` - Candidate details
- ✅ `/api/assess-audio` - Upload audio → get scores
- ✅ `/api/health` - Health check

## 🎯 Usage

1. **Visit Home Page**: http://localhost:5173
2. **Try Live Assessment**: Click "Try Live Assessment"
3. **Record Audio**: Click microphone, speak for 30-60 seconds
4. **View Results**: See C3 scores with gauge charts
5. **Browse Candidates**: Explore 44 pre-assessed candidates

## 🔧 Troubleshooting

### Backend Error: "Module not found"
```bash
cd backend
pip install Flask Flask-CORS pandas numpy scikit-learn
```

### Frontend Error: "npm: command not found"
Install Node.js from https://nodejs.org

### CORS Error
Make sure Flask backend is running on port 5000

### Audio Recording Not Working
- Allow microphone permissions in browser
- Use HTTPS or localhost only

## 📊 API Endpoints

```
GET  /api/health           - Health check
GET  /api/candidates       - List all candidates
GET  /api/candidate/:id    - Get candidate details
POST /api/assess-audio     - Upload audio file
```

## 🎨 Design System

- **Primary Color**: Blue (#2563eb)
- **Fonts**: Inter (Google Fonts)
- **Components**: Tailwind CSS utility classes
- **Icons**: Lucide React

## 🚢 Deployment

### Frontend (Vercel/Netlify)
```bash
cd frontend
npm run build
# Deploy dist/ folder
```

### Backend (Heroku/Railway)
```bash
cd backend
# Add Procfile: web: gunicorn app:app
```

## ⏱️ Build Time

- Backend: ~30 minutes
- Frontend: ~8 hours
- **Total**: ~8.5 hours

## 📝 Notes

- Streamlit app still available: `streamlit run app.py`
- React app is production-ready
- Backend needs models and data from parent directory
