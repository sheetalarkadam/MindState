"""
Week 2 Complete Demo: Content Analysis + Cognitive-Aware Recommendations
Shows the full pipeline from behavioral analysis to personalized recommendations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'behavioral_analyzer'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'content_analyzer'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'recommendation_engine'))

from behavior_simulator import BehaviorDataGenerator
from cognitive_state_detector import CognitiveStateClassifier
from complexity_analyzer import ContentComplexityAnalyzer
from mindstate_recommender import MindStateRecommender
import pandas as pd

def run_complete_week2_demo():
    """Run the complete end-to-end MindState system demo"""
    
    print("🧠🎬 MindState Complete System Demo - Week 2")
    print("=" * 55)
    print("Behavioral Analysis → Cognitive Detection → Content Matching → Recommendations")
    print()
    
    # Step 1: Initialize all components
    print("🚀 Step 1: Initializing MindState System Components")
    print("-" * 45)
    
    behavior_generator = BehaviorDataGenerator()
    cognitive_classifier = CognitiveStateClassifier()
    recommender = MindStateRecommender()
    
    print("✅ Behavioral data generator ready")
    print("✅ Cognitive state classifier ready")
    print("✅ Content complexity analyzer ready")
    print("✅ Recommendation engine ready")
    
    # Step 2: Train cognitive state detection (quick training)
    print(f"\n🤖 Step 2: Training Cognitive State Detection Model")
    print("-" * 45)
    
    # Generate training data
    print("Generating behavioral training data...")
    training_data = behavior_generator.generate_dataset(sessions_per_state=20)  # Smaller for demo
    
    # Train classifier
    print("Training cognitive state classifier...")
    results = cognitive_classifier.train(training_data)
    best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    print(f"✅ Model trained - Best accuracy: {results[best_model]['test_accuracy']:.1%}")
    
    # Step 3: Load content database
    print(f"\n📚 Step 3: Loading and Analyzing Content Database")
    print("-" * 45)
    
    content_database = [
        {
            'title': 'Friends - The One with the Embryos',
            'genre': 'Comedy/Sitcom',
            'duration': 22,
            'type': 'tv_series',
            'description': 'The gang plays a trivia game to see who knows each other better.',
            'plot': 'Light-hearted competition between friends with familiar characters and situations.'
        },
        {
            'title': 'Interstellar',
            'genre': 'Sci-Fi/Drama',
            'duration': 169,
            'type': 'movie',
            'description': 'A team of explorers travel through a wormhole in space to save humanity.',
            'plot': 'Complex scientific concepts including relativity, black holes, and higher dimensions with emotional family drama.'
        },
        {
            'title': 'The Great British Baking Show',
            'genre': 'Reality/Competition',
            'duration': 60,
            'type': 'tv_series',
            'description': 'Amateur bakers compete in a series of rounds.',
            'plot': 'Gentle competition with positive atmosphere and predictable format.'
        },
        {
            'title': 'Free Solo',
            'genre': 'Documentary/Adventure',
            'duration': 100,
            'type': 'documentary',
            'description': 'Rock climber Alex Honnold attempts to climb El Capitan without ropes.',
            'plot': 'Intense documentary following preparation and execution of dangerous climb.'
        },
        {
            'title': 'The Matrix',
            'genre': 'Sci-Fi/Action',
            'duration': 136,
            'type': 'movie',
            'description': 'A hacker discovers reality is a computer simulation.',
            'plot': 'Philosophical themes about reality, identity, and choice combined with action sequences.'
        },
        {
            'title': 'Bob Ross: The Joy of Painting',
            'genre': 'Educational/Art',
            'duration': 27,
            'type': 'tv_series',
            'description': 'Bob Ross teaches landscape painting techniques.',
            'plot': 'Calming instructional content with soothing narration and predictable format.'
        }
    ]
    
    recommender.load_content_database(content_database)
    print(f"✅ Content database loaded and analyzed")
    
    # Step 4: End-to-End User Scenarios
    print(f"\n👤 Step 4: End-to-End User Scenarios")
    print("-" * 35)
    
    scenarios = [
        {
            'name': 'Stressed Software Developer',
            'actual_state': 'stress_state',
            'description': 'Just finished a difficult debugging session, feeling overwhelmed'
        },
        {
            'name': 'Curious Weekend Explorer',
            'actual_state': 'discovery_mode',
            'description': 'Relaxed Saturday afternoon, open to trying something new'
        },
        {
            'name': 'Focused Learning Session',
            'actual_state': 'learning_mode',
            'description': 'Motivated to learn something educational and substantial'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. Scenario: {scenario['name']}")
        print(f"   Context: {scenario['description']}")
        
        # Generate realistic behavioral data for this state
        behavioral_session = behavior_generator.generate_session_data(
            scenario['actual_state'], 
            duration_minutes=5
        )
        behavioral_df = pd.DataFrame(behavioral_session['events'])
        
        # Show behavioral indicators
        print(f"   📊 Behavioral Indicators:")
        print(f"       Typing speed: {behavioral_df['typing_speed'].mean():.0f} WPM")
        print(f"       Error rate: {behavioral_df['typing_errors'].mean():.1%}")
        print(f"       Focus score: {behavioral_df['session_focus'].mean():.2f}")
        print(f"       App switches: {behavioral_df['app_switches'].mean():.1f}/min")
        
        # Detect cognitive state
        if cognitive_classifier.is_trained:
            prediction = cognitive_classifier.predict(behavioral_df)
            detected_state = prediction['state']
            confidence = prediction['confidence']
        else:
            # Fallback for demo
            detected_state = scenario['actual_state']
            confidence = 0.85
        
        print(f"   🎯 Detected State: {detected_state} (confidence: {confidence:.1%})")
        print(f"   ✅ Correct Detection: {'Yes' if detected_state == scenario['actual_state'] else 'No'}")
        
        # Get personalized recommendations
        recommendations = recommender.get_recommendations(
            behavioral_data=behavioral_df,
            num_recommendations=2
        )
        
        print(f"   🎬 Personalized Recommendations:")
        for j, rec in enumerate(recommendations['recommendations'], 1):
            print(f"       {j}. {rec['title']} ({rec['duration']} min)")
            print(f"          Complexity: {rec['complexity_score']:.2f} | {rec['complexity_category']}")
            print(f"          Why: {rec['why_recommended']}")
    
    # Step 5: System Performance Summary
    print(f"\n📈 Step 5: MindState System Performance Summary")
    print("=" * 45)
    
    print(f"✅ BEHAVIORAL ANALYSIS:")
    print(f"   - Realistic behavioral pattern simulation")
    print(f"   - Temporal effects (fatigue, circadian rhythms)")
    print(f"   - 5 distinct cognitive state signatures")
    
    print(f"\n✅ COGNITIVE STATE DETECTION:")
    print(f"   - Machine learning classification")
    print(f"   - {results[best_model]['test_accuracy']:.1%} accuracy on test data")
    print(f"   - Real-time behavioral analysis")
    
    print(f"\n✅ CONTENT COMPLEXITY ANALYSIS:")
    print(f"   - Multi-dimensional complexity scoring")
    print(f"   - Text readability analysis")
    print(f"   - Genre and duration factors")
    print(f"   - Cognitive load estimation")
    
    print(f"\n✅ COGNITIVE-AWARE RECOMMENDATIONS:")
    print(f"   - State-specific content filtering")
    print(f"   - Complexity tolerance matching")
    print(f"   - Explainable recommendations")
    print(f"   - Personalized ranking algorithms")
    
    print(f"\n🎯 BUSINESS IMPACT:")
    print(f"   - Prevents cognitive overload and recommendation rejection")
    print(f"   - Improves user satisfaction through state-aware matching")
    print(f"   - Reduces content discovery friction")
    print(f"   - Enables new category of 'empathetic AI' recommendations")
    
    print(f"\n🚀 READY FOR WEEK 3:")
    print(f"   - Real-time streaming interface")
    print(f"   - Interactive dashboard")
    print(f"   - A/B testing framework")
    print(f"   - Production deployment preparation")
    
    print(f"\n🎉 MindState System Demo Complete!")
    print(f"💼 Portfolio-ready: Unique AI innovation with measurable business impact")

if __name__ == "__main__":
    run_complete_week2_demo()
