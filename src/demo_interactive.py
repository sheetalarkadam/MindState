# Create the Streamlit interactive demo
"""
Interactive MindState Demo with Streamlit
Beautiful interactive interface for demonstrating the complete system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
import os
from datetime import datetime, timedelta

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'behavioral_analyzer'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'content_analyzer'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'recommendation_engine'))

from behavior_simulator import BehaviorDataGenerator
from cognitive_state_detector import CognitiveStateClassifier
from mindstate_recommender import MindStateRecommender

def init_session_state():
    """Initialize Streamlit session state"""
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
        st.session_state.behavior_generator = BehaviorDataGenerator()
        st.session_state.cognitive_classifier = CognitiveStateClassifier()
        st.session_state.recommender = MindStateRecommender()
        st.session_state.behavioral_history = []
        st.session_state.state_history = []
        st.session_state.session_start = datetime.now()

def train_system():
    """Train the MindState system"""
    if not st.session_state.system_initialized:
        with st.spinner('Training MindState cognitive detection system...'):
            # Generate training data
            training_data = st.session_state.behavior_generator.generate_dataset(sessions_per_state=25)
            
            # Train classifier
            results = st.session_state.cognitive_classifier.train(training_data)
            
            # Setup content database
            content_db = [
                {
                    'title': 'Friends - Sitcom Classic',
                    'genre': 'Comedy/Sitcom',
                    'duration': 22,
                    'type': 'tv_series',
                    'description': 'Six friends navigate life in NYC.',
                    'plot': 'Light-hearted, familiar situations with beloved characters.'
                },
                {
                    'title': 'The Office - Workplace Comedy',
                    'genre': 'Comedy/Mockumentary',
                    'duration': 22,
                    'type': 'tv_series',
                    'description': 'Documentary-style workplace comedy.',
                    'plot': 'Character-driven humor with minimal cognitive load.'
                },
                {
                    'title': 'Inception - Mind-Bending Thriller',
                    'genre': 'Sci-Fi/Thriller',
                    'duration': 148,
                    'type': 'movie',
                    'description': 'Dreams within dreams heist thriller.',
                    'plot': 'Complex multi-layered narrative requiring intense focus.'
                },
                {
                    'title': 'Planet Earth - Nature Documentary',
                    'genre': 'Documentary/Nature',
                    'duration': 50,
                    'type': 'documentary',
                    'description': 'Stunning wildlife documentary.',
                    'plot': 'Educational content with beautiful visuals.'
                },
                {
                    'title': 'Great British Baking Show',
                    'genre': 'Reality/Competition',
                    'duration': 60,
                    'type': 'tv_series',
                    'description': 'Gentle baking competition.',
                    'plot': 'Calming, predictable format perfect for relaxation.'
                },
                {
                    'title': 'Cosmos - Science Documentary',
                    'genre': 'Documentary/Science',
                    'duration': 43,
                    'type': 'documentary',
                    'description': 'Space and scientific exploration.',
                    'plot': 'Complex scientific concepts requiring focused attention.'
                }
            ]
            
            st.session_state.recommender.load_content_database(content_db)
            st.session_state.system_initialized = True
            
        st.success('✅ MindState system trained and ready!')

def create_behavioral_metrics_chart(behavioral_data):
    """Create real-time behavioral metrics visualization"""
    
    if not behavioral_data:
        return go.Figure()
    
    df = pd.DataFrame(behavioral_data)
    df['time'] = range(len(df))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Typing Speed', 'Focus Score', 'Mouse Precision', 'App Switches'),
        vertical_spacing=0.1
    )
    
    # Typing Speed
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['typing_speed'], 
                  mode='lines+markers', name='Typing Speed',
                  line=dict(color='#1f77b4', width=3)),
        row=1, col=1
    )
    
    # Focus Score
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['session_focus'], 
                  mode='lines+markers', name='Focus Score',
                  line=dict(color='#ff7f0e', width=3)),
        row=1, col=2
    )

# Mouse Precision
    fig.add_trace(
       go.Scatter(x=df['time'], y=df['mouse_precision'], 
                 mode='lines+markers', name='Mouse Precision',
                 line=dict(color='#2ca02c', width=3)),
       row=2, col=1
   )
   
   # App Switches
    fig.add_trace(
       go.Scatter(x=df['time'], y=df['app_switches'], 
                 mode='lines+markers', name='App Switches',
                 line=dict(color='#d62728', width=3)),
       row=2, col=2
   )
   
    fig.update_layout(
       height=400,
       showlegend=False,
       title_text="Real-Time Behavioral Metrics",
       title_x=0.5
   )
   
    return fig

def create_state_timeline_chart(state_history):
   """Create cognitive state timeline visualization"""
   
   if not state_history:
       return go.Figure()
   
   df = pd.DataFrame(state_history)
   
   # Map states to colors
   state_colors = {
       'stress_state': '#d62728',
       'recovery_mode': '#ff7f0e', 
       'discovery_mode': '#2ca02c',
       'learning_mode': '#1f77b4',
       'flow_state': '#9467bd'
   }
   
   colors = [state_colors.get(state, '#gray') for state in df['state']]
   
   fig = go.Figure()
   
   fig.add_trace(go.Scatter(
       x=df['time'],
       y=df['confidence'],
       mode='markers+lines',
       marker=dict(
           size=12,
           color=colors,
           line=dict(width=2, color='white')
       ),
       line=dict(width=3),
       text=df['state'],
       hovertemplate='<b>%{text}</b><br>Confidence: %{y:.1%}<br>Time: %{x}<extra></extra>'
   ))
   
   fig.update_layout(
       title="Cognitive State Timeline",
       xaxis_title="Time",
       yaxis_title="Confidence",
       height=300,
       yaxis=dict(tickformat=".0%")
   )
   
   return fig

def main():
   """Main Streamlit application"""
   
   st.set_page_config(
       page_title="MindState Demo",
       page_icon="🧠",
       layout="wide",
       initial_sidebar_state="expanded"
   )
   
   # Initialize session state
   init_session_state()
   
   # Header
   st.title("🧠 MindState: Cognitive-Aware Recommendations")
   st.markdown("*The first recommendation system that adapts to your mental state and cognitive load*")
   
   # Sidebar controls
   st.sidebar.header("🎛️ Demo Controls")
   
   # Train system button
   if not st.session_state.system_initialized:
       if st.sidebar.button("🚀 Initialize MindState System", type="primary"):
           train_system()
   else:
       st.sidebar.success("✅ System Ready")
   
   # Manual state override
   manual_state = st.sidebar.selectbox(
       "Override Cognitive State (for demo):",
       ["Auto-detect", "stress_state", "recovery_mode", "discovery_mode", "learning_mode", "flow_state"]
   )
   
   # Real-time simulation controls
   st.sidebar.subheader("⏱️ Real-Time Simulation")
   auto_update = st.sidebar.checkbox("Auto-update every 2 seconds", value=False)
   
   if st.sidebar.button("Generate New Behavioral Event"):
       generate_behavioral_event(manual_state)
   
   if st.sidebar.button("🔄 Reset Session"):
       st.session_state.behavioral_history = []
       st.session_state.state_history = []
       st.session_state.session_start = datetime.now()
       st.rerun()
   
   # Main content area
   if not st.session_state.system_initialized:
       st.info("👆 Click 'Initialize MindState System' in the sidebar to begin the demo")
       return
   
   # Create main dashboard
   col1, col2 = st.columns([2, 1])
   
   with col1:
       st.subheader("📊 Real-Time Behavioral Analysis")
       
       # Behavioral metrics chart
       behavioral_chart = create_behavioral_metrics_chart(st.session_state.behavioral_history)
       st.plotly_chart(behavioral_chart, use_container_width=True)
       
       # State timeline
       if st.session_state.state_history:
           st.subheader("🧠 Cognitive State Detection Timeline")
           state_chart = create_state_timeline_chart(st.session_state.state_history)
           st.plotly_chart(state_chart, use_container_width=True)
   
   with col2:
       st.subheader("🎯 Current Status")
       
       # Current metrics
       if st.session_state.behavioral_history:
           latest = st.session_state.behavioral_history[-1]
           
           # Metrics display
           col_a, col_b = st.columns(2)
           with col_a:
               st.metric("⌨️ Typing Speed", f"{latest['typing_speed']:.0f} WPM")
               st.metric("🎯 Focus Score", f"{latest['session_focus']:.2f}")
               st.metric("🖱️ Mouse Precision", f"{latest['mouse_precision']:.2f}")
           
           with col_b:
               st.metric("❌ Error Rate", f"{latest['typing_errors']:.1%}")
               st.metric("🔄 App Switches", f"{latest['app_switches']:.1f}/min")
               
               session_duration = (datetime.now() - st.session_state.session_start).total_seconds()
               st.metric("⏱️ Session Time", f"{session_duration:.0f}s")
       
       # Current cognitive state
       if st.session_state.state_history:
           latest_state = st.session_state.state_history[-1]
           
           st.subheader("🧠 Detected Cognitive State")
           
           # State display with confidence
           state_name = latest_state['state'].replace('_', ' ').title()
           confidence = latest_state['confidence']
           
           # Color coding for states
           state_colors = {
               'stress_state': '🔴',
               'recovery_mode': '🟡', 
               'discovery_mode': '🟢',
               'learning_mode': '🔵',
               'flow_state': '🟣'
           }
           
           state_emoji = state_colors.get(latest_state['state'], '⚪')
           
           st.markdown(f"""
           ### {state_emoji} {state_name}
           **Confidence:** {confidence:.1%}
           
           **State Description:**
           {get_state_description(latest_state['state'])}
           """)
           
           # Progress bar for confidence
           st.progress(confidence)
   
   # Recommendations section
   if st.session_state.behavioral_history and len(st.session_state.behavioral_history) >= 3:
       st.subheader("🎬 Personalized Recommendations")
       
       # Get recommendations
       latest_behavior_df = pd.DataFrame(st.session_state.behavioral_history[-10:])
       
       current_state = manual_state if manual_state != "Auto-detect" else None
       
       recommendations = st.session_state.recommender.get_recommendations(
           behavioral_data=latest_behavior_df if current_state is None else None,
           cognitive_state=current_state,
           num_recommendations=4
       )
       
       # Display recommendations
       st.markdown(f"""
       **Optimized for:** {recommendations['state_description']}  
       **Complexity Tolerance:** {recommendations['complexity_tolerance']:.1f}  
       **Available Options:** {recommendations['total_suitable_items']} items
       """)
       
       # Recommendation cards
       cols = st.columns(2)
       for i, rec in enumerate(recommendations['recommendations']):
           with cols[i % 2]:
               with st.container():
                   st.markdown(f"""
                   #### 🎬 {rec['title']}
                   **Genre:** {rec['genre']}  
                   **Duration:** {rec['duration']} minutes  
                   **Complexity:** {rec['complexity_category']} ({rec['complexity_score']:.2f})  
                   
                   **Why recommended:** {rec['why_recommended']}
                   
                   **Cognitive Load Info:**
                   - Attention span: {rec['cognitive_load_info']['attention_span_required']}
                   - Memory load: {rec['cognitive_load_info']['working_memory_load']}
                   - Processing speed: {rec['cognitive_load_info']['processing_speed_required']}
                   """)
                   st.markdown("---")
   
   # Auto-update functionality
   if auto_update and st.session_state.system_initialized:
       time.sleep(2)
       generate_behavioral_event(manual_state)
       st.rerun()

def generate_behavioral_event(manual_state):
   """Generate a new behavioral event and update state"""
   
   if not st.session_state.system_initialized:
       return
   
   # Calculate session duration
   session_duration = (datetime.now() - st.session_state.session_start).total_seconds()
   
   # Override generator state if manual mode
   if manual_state != "Auto-detect":
       st.session_state.behavior_generator.current_state = manual_state
   
   # Generate behavioral event
   event = st.session_state.behavior_generator.generate_realtime_event(session_duration)
   st.session_state.behavioral_history.append(event)
   
   # Keep history manageable
   if len(st.session_state.behavioral_history) > 50:
       st.session_state.behavioral_history.pop(0)
   
   # Predict cognitive state if we have enough data
   if len(st.session_state.behavioral_history) >= 5:
       behavior_df = pd.DataFrame(st.session_state.behavioral_history[-10:])
       
       try:
           prediction = st.session_state.cognitive_classifier.predict(behavior_df)
           
           state_event = {
               'time': len(st.session_state.state_history),
               'state': prediction['state'],
               'confidence': prediction['confidence'],
               'timestamp': datetime.now()
           }
           
           st.session_state.state_history.append(state_event)
           
           # Keep state history manageable
           if len(st.session_state.state_history) > 50:
               st.session_state.state_history.pop(0)
               
       except Exception as e:
           st.error(f"Prediction error: {e}")

def get_state_description(state):
   """Get description for cognitive state"""
   descriptions = {
       'stress_state': 'Feeling overwhelmed and need calming, simple content to reduce cognitive load.',
       'recovery_mode': 'Mentally tired and prefer familiar, easy-to-process content for relaxation.',
       'discovery_mode': 'Curious and ready to explore new content with moderate complexity.',
       'learning_mode': 'Focused on learning and can handle educational, substantial content.',
       'flow_state': 'In deep focus and can tackle complex, challenging content requiring sustained attention.'
   }
   return descriptions.get(state, 'Current cognitive state detected from behavioral patterns.')

if __name__ == "__main__":
   main()