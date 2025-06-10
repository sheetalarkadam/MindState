"""
LLM Enhanced MindState Demo
Compare traditional ML vs LLM approaches with rich explanations
"""

import streamlit as st
import pandas as pd
import asyncio
import sys
import os
import time

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'behavioral_analyzer'))

def main():
    st.set_page_config(page_title="MindState LLM Enhanced", layout="wide")
    
    st.title("�� MindState: LLM-Enhanced Cognitive Analysis")
    st.markdown("*Compare traditional ML with advanced LLM reasoning*")
    
    # Initialize components
    if 'components_loaded' not in st.session_state:
        load_components()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Demo Controls")
    
    test_state = st.sidebar.selectbox(
        "Generate Test Data For:",
        ['flow_state', 'stress_state', 'recovery_mode', 'learning_mode', 'discovery_mode']
    )
    
    if st.sidebar.button("🔄 Generate New Behavioral Data"):
        generate_test_data(test_state)
    
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode:",
        ["Compare Both Methods", "LLM Analysis Only", "Traditional ML Only"]
    )
    
    # Main interface
    if 'behavioral_data' not in st.session_state:
        st.info("👆 Generate behavioral data to start analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Behavioral Data")
        show_behavioral_metrics()
        
        if analysis_mode in ["Compare Both Methods", "Traditional ML Only"]:
            st.subheader("🔬 Traditional ML Analysis")
            run_traditional_analysis()
    
    with col2:
        if analysis_mode in ["Compare Both Methods", "LLM Analysis Only"]:
            st.subheader("🤖 LLM-Enhanced Analysis")
            run_llm_analysis()

def load_components():
    """Load all necessary components"""
    try:
        from behavior_simulator import BehaviorDataGenerator
        from cognitive_state_detector import CognitiveStateClassifier
        from llm_cognitive_detector import LLMCognitiveAnalyzer
        
        st.session_state.generator = BehaviorDataGenerator()
        st.session_state.traditional_classifier = CognitiveStateClassifier()
        st.session_state.llm_analyzer = LLMCognitiveAnalyzer()
        st.session_state.components_loaded = True
        
        # Quick train traditional classifier
        with st.spinner("Training traditional ML model..."):
            training_data = st.session_state.generator.generate_dataset(sessions_per_state=15)
            st.session_state.traditional_classifier.train(training_data)
        
    except Exception as e:
        st.error(f"Error loading components: {e}")

def generate_test_data(state):
    """Generate test behavioral data"""
    session = st.session_state.generator.generate_session_data(state, duration_minutes=3)
    st.session_state.behavioral_data = pd.DataFrame(session['events'])
    st.session_state.actual_state = state
    st.success(f"Generated behavioral data for {state}")

def show_behavioral_metrics():
    """Display current behavioral metrics"""
    if 'behavioral_data' not in st.session_state:
        return
    
    data = st.session_state.behavioral_data
    latest = data.iloc[-1]
    
    # Key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("⌨️ Typing Speed", f"{latest['typing_speed']:.0f} WPM")
        st.metric("🎯 Focus Score", f"{latest['session_focus']:.2f}")
        st.metric("🖱️ Mouse Precision", f"{latest['mouse_precision']:.2f}")
    
    with col2:
        st.metric("❌ Error Rate", f"{latest['typing_errors']:.1%}")
        st.metric("🔄 App Switches", f"{latest['app_switches']:.1f}/min")
        st.metric("📏 Session Length", f"{len(data)} data points")
    
    # Show trends
    if len(data) > 5:
        st.subheader("📈 Behavioral Trends")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data['typing_speed'], 
            mode='lines+markers',
            name='Typing Speed',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            y=data['session_focus'] * 100,  # Scale for visibility
            mode='lines+markers', 
            name='Focus Score (×100)',
            line=dict(color='green')
        ))
        
        fig.update_layout(height=200, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

def run_traditional_analysis():
    """Run traditional ML analysis"""
    
    try:
        # Predict with traditional ML
        prediction = st.session_state.traditional_classifier.predict(st.session_state.behavioral_data)
        
        # Display results
        st.metric("🔬 ML Prediction", prediction['state'])
        st.metric("📊 Confidence", f"{prediction['confidence']:.1%}")
        
        # Accuracy check
        actual_state = st.session_state.get('actual_state', 'unknown')
        is_correct = prediction['state'] == actual_state
        
        if is_correct:
            st.success(f"✅ Correct! (Actual: {actual_state})")
        else:
            st.error(f"❌ Incorrect (Actual: {actual_state})")
        
        # Show model details
        with st.expander("🔍 Traditional ML Details"):
            st.write("**Method:** Random Forest + Gradient Boosting")
            st.write("**Features:** Statistical aggregation of behavioral patterns")
            st.write("**Training:** Supervised learning on labeled behavioral data")
            
    except Exception as e:
        st.error(f"Traditional ML analysis failed: {e}")

def run_llm_analysis():
    """Run LLM-enhanced analysis"""
    
    try:
        with st.spinner("🤖 Analyzing with LLM..."):
            # Run async LLM analysis
            analysis = asyncio.run(
                st.session_state.llm_analyzer.predict_cognitive_state(st.session_state.behavioral_data)
            )
        
        # Display results
        st.metric("🤖 LLM Prediction", analysis['state'])
        st.metric("🎯 Confidence", f"{analysis['confidence']:.1%}")
        
        # Accuracy check
        actual_state = st.session_state.get('actual_state', 'unknown')
        is_correct = analysis['state'] == actual_state
        
        if is_correct:
            st.success(f"✅ Correct! (Actual: {actual_state})")
        else:
            st.error(f"❌ Incorrect (Actual: {actual_state})")
        
        # Show detailed LLM reasoning
        st.subheader("🧠 LLM Reasoning")
        st.write(analysis.get('reasoning', 'No reasoning provided'))
        
        # Show evidence
        if analysis.get('behavioral_evidence'):
            st.subheader("📋 Supporting Evidence")
            for evidence in analysis['behavioral_evidence']:
                st.write(f"• {evidence}")
        
        # Show temporal analysis
        if analysis.get('temporal_analysis'):
            st.subheader("⏰ Temporal Analysis")
            st.write(analysis['temporal_analysis'])
        
        # Show alternative states
        alt_states = analysis.get('alternative_states', {})
        if alt_states.get('second_most_likely'):
            st.subheader("🔄 Alternative Consideration")
            st.write(f"**{alt_states['second_most_likely']}** ({alt_states.get('probability', 0):.1%})")
            st.write(alt_states.get('why_considered', ''))
        
        # Content recommendations
        if analysis.get('content_recommendations'):
            st.subheader("�� Content Recommendations")
            st.write(analysis['content_recommendations'])
        
        # Show full explanation
        with st.expander("📖 Complete LLM Analysis"):
            explanation = st.session_state.llm_analyzer.explain_prediction(
                analysis, st.session_state.behavioral_data
            )
            st.markdown(explanation)
            
    except Exception as e:
        st.error(f"LLM analysis failed: {e}")
        st.info("💡 Make sure Ollama is running with llama2 model")

if __name__ == "__main__":
    main()
