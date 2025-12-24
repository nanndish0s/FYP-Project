"""
Live Audio Assessment Tab for Streamlit App
Adds microphone recording and real-time C3 assessment
"""
import streamlit as st
import os
import sys
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add to import section of app.py
sys.path.insert(0, '.')
from src.audio.record_audio import AudioRecorder
from src.audio.transcribe_audio import SpeechTranscriber
from src.features.extract_live_features import LiveFeatureExtractor

def live_assessment_tab():
    """
    Live Assessment Tab
    Allows candidates to record audio and get instant C3 assessment
    """
    st.header("🎤 Live Candidate Assessment")
    
    st.markdown("""
    ### Record Your Response
    **Instructions:**
    1. Click the button below to start recording
    2. Speak for 30-60 seconds about a technical topic
    3. System will automatically transcribe and assess your soft skills
    4. View your C3 scores with explanations
    
    **Suggested Topics:**
    - Describe a challenging project you've worked on
    - Explain how you would approach a system design problem
    - Discuss a time you had to learn a new technology quickly
    """)
    
    st.markdown("---")
    
    # Recording controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        duration = st.number_input("Recording Duration (seconds)", min_value=10, max_value=90, value=60)
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        record_button = st.button("🎤 Start Recording", type="primary", use_container_width=True)
    
    with col3:
        st.write("")  # Spacing
        st.write("")
        test_mic = st.button("🧪 Test Microphone", use_container_width=True)
    
    # Test microphone
    if test_mic:
        with st.spinner("Testing microphone..."):
            try:
                recorder = AudioRecorder()
                rms = recorder.test_microphone(duration=3)
                
                # Show actual level to user
                st.info(f"📊 Microphone Level: {rms:.0f} RMS")
                
                # Much more lenient thresholds
                if rms < 10:
                    st.error("❌ Microphone level too low! Please speak louder or check Windows microphone permissions.")
                    st.info("💡 **Troubleshooting**: Go to Windows Settings → Privacy → Microphone → Allow apps to access your microphone")
                elif rms > 5000:
                    st.warning("⚠️ Audio might be too loud (possible clipping)")
                else:
                    st.success(f"✅ Microphone working well! (Level: {rms:.0f})")
            except Exception as e:
                st.error(f"❌ Microphone test failed: {str(e)}")
        
    # Record and assess
    if record_button:
        try:
            # Initialize pipeline
            with st.spinner("Initializing..."):
                recorder = AudioRecorder(sample_rate=16000)
                transcriber = SpeechTranscriber(model_size="base")
                feature_extractor = LiveFeatureExtractor()
                
                # Load models
                models = {}
                for trait in ['curiosity', 'critical_thinking', 'creativity']:
                    model_file = f'models/{trait}_model_44.pkl'
                    if not os.path.exists(model_file):
                        model_file = f'models/{trait}_score_model.pkl'
                    with open(model_file, 'rb') as f:
                        models[trait] = pickle.load(f)
                
                # Load feature schema
                ml_dataset_path = 'data/processed/ml_ready_dataset_44.csv'
                if not os.path.exists(ml_dataset_path):
                    ml_dataset_path = 'data/processed/ml_ready_dataset.csv'
                df_train = pd.read_csv(ml_dataset_path)
                feature_cols = [col for col in df_train.columns 
                               if col not in ['video_id', 'transcript', 'word_count',
                                             'curiosity_score', 'critical_thinking_score', 
                                             'creativity_score', 'curiosity_reasoning',
                                             'critical_thinking_reasoning', 'creativity_reasoning']]
                feature_means = df_train[feature_cols].mean().to_dict()
            
            # Step 1: Record with live feedback
            st.info(f"🎤 **Recording for {duration} seconds...**")
            st.markdown("**Speak now!** Share your thoughts clearly and naturally.")
            
            # Prepare audio path
            audio_path = os.path.join("data/temp", "live_recording.wav")
            os.makedirs("data/temp", exist_ok=True)
            
            # Start recording in background and show timer
            import threading
            import time
            
            recording_complete = threading.Event()
            
            def do_recording():
                recorder.record(duration=duration, output_path=audio_path)
                recording_complete.set()
            
            # Start recording thread
            record_thread = threading.Thread(target=do_recording)
            record_thread.start()
            
            # Show live countdown while recording
            timer_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            for elapsed in range(duration + 1):
                remaining = duration - elapsed
                if remaining >= 0:
                    timer_placeholder.metric(
                        "⏱️ Recording Progress", 
                        f"{elapsed}/{duration} seconds",
                        delta=f"{remaining}s remaining"
                    )
                    progress_bar.progress(min(elapsed / duration, 1.0))
                time.sleep(1)
            
            # Wait for recording to complete
            recording_complete.wait()
            record_thread.join()
            
            timer_placeholder.empty()
            audio_file = audio_path
            
            progress_bar.progress(0.33)
            st.success("✅ Recording complete!")
            
            # Step 2: Transcribe
            with st.spinner("📝 Transcribing speech..."):
                transcript = transcriber.transcribe(audio_file)
                progress_bar.progress(66)
            
            st.success("✅ Transcription complete!")
            
            # Display transcript
            st.subheader("📄 Your Transcript")
            st.write(transcript)
            st.caption(f"Word count: {len(transcript.split())} words")
            
            # Step 3: Extract features
            with st.spinner("🔬 Extracting features..."):
                features = feature_extractor.extract_all_features(audio_file, transcript)
                
                # Prepare feature vector
                feature_vector = []
                for feature_name in feature_cols:
                    if feature_name in features:
                        feature_vector.append(features[feature_name])
                    else:
                        feature_vector.append(feature_means.get(feature_name, 0))
                feature_vector = np.array(feature_vector).reshape(1, -1)
                progress_bar.progress(90)
            
            # Step 4: Predict
            with st.spinner("🎯 Predicting C3 scores..."):
                predictions = {}
                for trait_name, model in models.items():
                    score = model.predict(feature_vector)[0]
                    score = np.clip(score, 1, 5)  # Ensure valid range
                    predictions[trait_name] = score
                progress_bar.progress(100)
            
            st.success("✅ Assessment complete!")
            
            # Display results
            st.markdown("---")
            st.header("🎯 Your C3 Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            def create_gauge_simple(value, title):
                """Create gauge chart"""
                if value >= 4.0:
                    color = "#28a745"
                elif value >= 3.5:
                    color = "#17a2b8"
                elif value >= 3.0:
                    color = "#ffc107"
                else:
                    color = "#dc3545"
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': title, 'font': {'size': 20}},
                    number={'suffix': "/5", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [None, 5]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 2], 'color': '#ffcccc'},
                            {'range': [2, 3], 'color': '#fff3cd'},
                            {'range': [3, 4], 'color': '#d1ecf1'},
                            {'range': [4, 5], 'color': '#d4edda'}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                return fig
            
            with col1:
                fig = create_gauge_simple(predictions['curiosity'], "Curiosity")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_gauge_simple(predictions['critical_thinking'], "Critical Thinking")
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = create_gauge_simple(predictions['creativity'], "Creativity")
                st.plotly_chart(fig, use_container_width=True)
            
            # Overall recommendation
            avg_score = np.mean(list(predictions.values()))
            
            st.markdown("---")
            st.subheader("💼 Assessment Summary")
            
            if avg_score >= 4.0:
                st.success(f"🌟 **EXCELLENT** - Overall Score: {avg_score:.2f}/5")
                st.markdown("Strong performance across all soft skills!")
            elif avg_score >= 3.5:
                st.success(f"✅ **GOOD** - Overall Score: {avg_score:.2f}/5")
                st.markdown("Solid soft skills demonstration.")
            elif avg_score >= 3.0:
                st.warning(f"⚠️ **MODERATE** - Overall Score: {avg_score:.2f}/5")
                st.markdown("Room for improvement in some areas.")
            else:
                st.error(f"❌ **NEEDS DEVELOPMENT** - Overall Score: {avg_score:.2f}/5")
                st.markdown("Consider developing soft skills further.")
            
            # Strengths and areas
            strengths = [name.replace('_', ' ').title() for name, score in predictions.items() if score >= 4.0]
            development = [name.replace('_', ' ').title() for name, score in predictions.items() if score < 3.5]
            
            col1, col2 = st.columns(2)
            with col1:
                if strengths:
                    st.markdown("**✅ Strengths:**")
                    for strength in strengths:
                        st.markdown(f"- {strength}")
            
            with col2:
                if development:
                    st.markdown("**📈 Growth Areas:**")
                    for area in development:
                        st.markdown(f"- {area}")
            
            # Note about feature extraction
            st.info("""
            **ℹ️ Note:** This assessment uses simplified acoustic feature extraction for live recordings. 
            Results may differ slightly from pre-recorded candidates due to feature approximation.
            """)
            
        except Exception as e:
            st.error(f"❌ Error during assessment: {str(e)}")
            st.exception(e)
