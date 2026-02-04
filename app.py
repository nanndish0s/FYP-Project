"""
Voice-Based AI Recruitment - Streamlit Web Application
Interactive demo for Final Year Project presentation
NOW WITH LIVE AUDIO ASSESSMENT!
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import sys

# Import live assessment module
sys.path.insert(0, '.')
from src.app_live_tab import live_assessment_tab

# Page config
st.set_page_config(
    page_title="Voice-Based AI Recruitment",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .score-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .score-good {
        color: #17a2b8;
        font-weight: bold;
    }
    .score-moderate {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎤 Voice-Based AI Recruitment System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable Soft Skills Assessment for Software Engineering Candidates</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📊 Pre-Recorded Candidates", "🎤 Live Assessment"])

with tab2:
    # Live assessment tab
    live_assessment_tab()

with tab1:
    # Cache data loading
    @st.cache_data
    def load_data():
        """Load all necessary data (Prioritize RecruitView)"""
        # Feature dataset (37 features + video_id)
        df_full_path = 'data/processed/recruitview_features_all.csv'
        if not os.path.exists(df_full_path):
            df_full_path = 'data/processed/ml_ready_dataset.csv'
            
        # Metadata dataset (Transcript + Target scores)
        df_labels_path = 'data/processed/recruitview_metadata.csv'
        if not os.path.exists(df_labels_path):
            df_labels_path = 'data/processed/sample_with_c3_labels.csv'
            
        df_full = pd.read_csv(df_full_path)
        df_labels = pd.read_csv(df_labels_path)
        
        # Ensure column consistency for metadata
        if 'curiosity' in df_labels.columns and 'curiosity_score' not in df_labels.columns:
            df_labels = df_labels.rename(columns={
                'curiosity': 'curiosity_score',
                'critical_thinking': 'critical_thinking_score',
                'creativity': 'creativity_score'
            })
            
        df_prosody = pd.read_csv('results/prosody_contributions.csv')
        return df_full, df_labels, df_prosody

    @st.cache_resource
    def load_models():
        """Load trained models"""
        models = {}
        for trait in ['curiosity', 'critical_thinking', 'creativity']:
            # Try to load RecruitView model, fall back to 44-sample, then original
            model_file = f'models/{trait}_model_recruitview.pkl'
            if not os.path.exists(model_file):
                model_file = f'models/{trait}_model_44.pkl'
                if not os.path.exists(model_file):
                    model_file = f'models/{trait}_score_model.pkl'
            
            with open(model_file, 'rb') as f:
                models[trait + '_score'] = pickle.load(f)
        
        return models

    def create_gauge(value, title, max_value=5):
        """Create a gauge chart for score visualization"""
        # Determine color based on value
        if value >= 4.0:
            color = "#28a745"  # Green
        elif value >= 3.5:
            color = "#17a2b8"  # Blue
        elif value >= 3.0:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 20}},
            number = {'suffix': "/5", 'font': {'size': 40}},
            gauge = {
                'axis': {'range': [None, max_value], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 2], 'color': '#ffcccc'},
                    {'range': [2, 3], 'color': '#fff3cd'},
                    {'range': [3, 4], 'color': '#d1ecf1'},
                    {'range': [4, 5], 'color': '#d4edda'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 3.5
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return fig

    def get_score_class(score):
        """Get CSS class for score coloring"""
        if score >= 4.0:
            return "score-excellent"
        elif score >= 3.5:
            return "score-good"
        elif score >= 3.0:
            return "score-moderate"
        else:
            return "score-low"

    # Load data
    with st.spinner('🔄 Loading models and data...'):
        df_full, df_labels, df_prosody = load_data()
        models = load_models()
        
        # Align with RecruitView 37-feature schema
        exclude_cols = ['video_id', 'curiosity_score', 'critical_thinking_score', 'creativity_score', 'file_name']
        feature_cols = [col for col in df_full.columns if col not in exclude_cols]

    # Sidebar
    st.sidebar.title("📋 Candidate Selection")
    st.sidebar.markdown("---")

    # Candidate selector
    candidate_options = [f"Candidate {i+1} - {row['video_id'][:15]}... ({row['word_count']} words)" 
                         for i, row in df_labels.iterrows()]
    selected_index = st.sidebar.selectbox(
        "Select a candidate to analyze:",
        range(len(candidate_options)),
        format_func=lambda x: candidate_options[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About This System:**
    
    This AI system assesses three critical soft skills for software engineering:
    - 🔍 **Curiosity**: Inquisitiveness, learning drive
    - 🧠 **Critical Thinking**: Analysis, problem-solving
    - 💡 **Creativity**: Innovation, novel solutions
    
    Using voice and speech analysis powered by advanced machine learning.
    """)

    # Get candidate data
    candidate_data = df_labels.iloc[selected_index]
    candidate_features = df_full.iloc[selected_index]
    X_sample = candidate_features[feature_cols].values.reshape(1, -1)

    # Main content
    st.markdown("---")

    # Section 1: Candidate Overview
    st.header("📝 Candidate Overview")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("Interview Transcript")
        with st.expander("📄 View Full Transcript", expanded=True):
            st.write(candidate_data['transcript'])

    with col2:
        st.metric("Word Count", f"{candidate_data['word_count']} words")
        st.metric("Sentence Count", f"{int(candidate_features['sentence_count'])}")

    with col3:
        st.metric("Vocabulary Diversity", f"{candidate_features['vocab_diversity']:.2f}")
        st.metric("Filler Word Ratio", f"{candidate_features['filler_word_ratio']:.1%}")

    st.markdown("---")

    # Section 2: C3 Predictions
    st.header("🤖 C3 Soft Skills Assessment")

    # Make predictions
    predictions = {}
    for trait in ['curiosity_score', 'critical_thinking_score', 'creativity_score']:
        pred = models[trait].predict(X_sample)[0]
        actual = candidate_data[trait]
        predictions[trait] = {'predicted': pred, 'actual': actual}

    # Display gauges
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = create_gauge(predictions['curiosity_score']['predicted'], "Curiosity")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Actual: {predictions['curiosity_score']['actual']:.0f}/5 | Error: ±{abs(predictions['curiosity_score']['predicted'] - predictions['curiosity_score']['actual']):.2f}")

    with col2:
        fig = create_gauge(predictions['critical_thinking_score']['predicted'], "Critical Thinking")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Actual: {predictions['critical_thinking_score']['actual']:.0f}/5 | Error: ±{abs(predictions['critical_thinking_score']['predicted'] - predictions['critical_thinking_score']['actual']):.2f}")

    with col3:
        fig = create_gauge(predictions['creativity_score']['predicted'], "Creativity")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Actual: {predictions['creativity_score']['actual']:.0f}/5 | Error: ±{abs(predictions['creativity_score']['predicted'] - predictions['creativity_score']['actual']):.2f}")

    st.markdown("---")

    st.markdown("---")

    # Section 5: Recruitment Recommendation
    st.header("💼 Recruitment Recommendation")

    avg_score = np.mean([predictions[t]['predicted'] for t in predictions])

    col1, col2 = st.columns([2, 1])

    with col1:
        if avg_score >= 4.0:
            st.success("🌟 **STRONG HIRE** - Excellent soft skills across all dimensions")
            recommendation_color = "green"
        elif avg_score >= 3.5:
            st.success("✅ **RECOMMENDED** - Good soft skills, suitable for role")
            recommendation_color = "blue"
        elif avg_score >= 3.0:
            st.warning("⚠️ **CONSIDER** - Moderate soft skills, may need development")
            recommendation_color = "orange"
        else:
            st.error("❌ **NOT RECOMMENDED** - Soft skills below threshold")
            recommendation_color = "red"
        
        st.markdown(f"**Average C3 Score:** `{avg_score:.2f}/5`")
        
        # Strengths
        strengths = [name.replace('_score', '').title() for name, data in predictions.items() if data['predicted'] >= 4.0]
        if strengths:
            st.markdown("**Strengths:**")
            for strength in strengths:
                st.markdown(f"- ✓ {strength}")
        
        # Development areas
        development = [name.replace('_score', '').title() for name, data in predictions.items() if data['predicted'] < 3.5]
        if development:
            st.markdown("**Areas for Development:**")
            for area in development:
                st.markdown(f"- ⚠ {area}")

    with col2:
        # Overall score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Score", 'font': {'size': 18}},
            number={'suffix': "/5", 'font': {'size': 35}},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': recommendation_color},
                'steps': [
                    {'range': [0, 3], 'color': 'lightgray'},
                    {'range': [3, 4], 'color': 'lightyellow'},
                    {'range': [4, 5], 'color': 'lightgreen'}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p><strong>Voice-Based AI Recruitment</strong></p>
    <p>Final Year Project | Software Engineering Soft Skills Assessment</p>
    <p>Using Random Forest for Transparent, Fair Hiring Decisions</p>
</div>
""", unsafe_allow_html=True)
