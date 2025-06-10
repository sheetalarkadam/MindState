"""
Week 1 Complete Demo: Behavioral Analysis + Cognitive State Detection
Shows the full pipeline from data generation to state prediction
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'behavioral_analyzer'))

from behavior_simulator import BehaviorDataGenerator
from cognitive_state_detector import CognitiveStateClassifier
import pandas as pd

def run_complete_week1_demo():
    """Run the complete Week 1 demonstration"""
    
    print("🧠 MindState Week 1 Complete Demo")
    print("=" * 50)
    print("Behavioral Analysis + Cognitive State Detection")
    print()
    
    # Step 1: Generate behavioral data
    print("📊 Step 1: Generating Behavioral Training Data")
    print("-" * 40)
    
    generator = BehaviorDataGenerator()
    
    # Show patterns for different states
    print("Behavioral patterns by cognitive state:")
    for state in ['flow_state', 'stress_state', 'recovery_mode']:
        sample_session = generator.generate_session_data(state, duration_minutes=3)
        sample_df = pd.DataFrame(sample_session['events'])
        
        avg_speed = sample_df['typing_speed'].mean()
        avg_errors = sample_df['typing_errors'].mean()
        avg_focus = sample_df['session_focus'].mean()
        avg_switches = sample_df['app_switches'].mean()
        
        print(f"  {state}:")
        print(f"    Typing: {avg_speed:.0f} WPM, {avg_errors:.1%} errors")
        print(f"    Focus: {avg_focus:.2f}, App switches: {avg_switches:.2f}/min")
    
    # Generate full training dataset
    print(f"\nGenerating full training dataset...")
    training_data = generator.generate_dataset(sessions_per_state=40)
    print(f"Created {len(training_data)} behavioral events across {training_data['session_id'].nunique()} sessions")
    
    # Step 2: Train cognitive state classifier
    print(f"\n🤖 Step 2: Training Cognitive State Classifier")
    print("-" * 40)
    
    classifier = CognitiveStateClassifier()
    results = classifier.train(training_data)
    
    # Show best performance
    best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_accuracy = results[best_model]['test_accuracy']
    print(f"\nBest model: {best_model} with {best_accuracy:.1%} accuracy")
    
    # Step 3: Real-time prediction simulation
    print(f"\n🔄 Step 3: Real-time Cognitive State Prediction")
    print("-" * 40)
    
    print("Simulating real-time behavioral monitoring...")
    
    # Simulate different scenarios
    scenarios = [
        ('flow_state', "User in deep focus session"),
        ('stress_state', "User under deadline pressure"), 
        ('recovery_mode', "User taking a mental break"),
        ('learning_mode', "User studying new material"),
        ('discovery_mode', "User exploring new content")
    ]
    
    correct_predictions = 0
    total_predictions = len(scenarios)
    
    for actual_state, description in scenarios:
        print(f"\n  Scenario: {description}")
        
        # Generate behavioral session for this state
        session = generator.generate_session_data(actual_state, duration_minutes=4)
        session_df = pd.DataFrame(session['events'])
        
        # Predict cognitive state
        prediction = classifier.predict(session_df)
        predicted_state = prediction['state']
        confidence = prediction['confidence']
        
        print(f"    Behavioral indicators:")
        print(f"      Typing speed: {session_df['typing_speed'].mean():.0f} WPM")
        print(f"      Error rate: {session_df['typing_errors'].mean():.1%}")
        print(f"      Focus score: {session_df['session_focus'].mean():.2f}")
        print(f"      App switches: {session_df['app_switches'].mean():.1f}/min")
        
        print(f"    🎯 Prediction: {predicted_state} (confidence: {confidence:.1%})")
        print(f"    ✅ Actual: {actual_state}")
        
        if predicted_state == actual_state:
            print(f"    ✅ CORRECT!")
            correct_predictions += 1
        else:
            print(f"    ❌ Incorrect")
    
    # Final results
    print(f"\n📈 Week 1 Results Summary")
    print("=" * 30)
    print(f"✅ Behavioral data generation: Working")
    print(f"✅ Cognitive state classification: {best_accuracy:.1%} accuracy")
    print(f"✅ Real-time prediction: {correct_predictions}/{total_predictions} correct")
    
    accuracy_percentage = (correct_predictions / total_predictions) * 100
    print(f"✅ End-to-end system accuracy: {accuracy_percentage:.0f}%")
    
    print(f"\n🚀 Week 1 Complete! Ready for Week 2:")
    print(f"   - Content complexity analysis")
    print(f"   - Cognitive-aware recommendation engine")
    print(f"   - Real-time streaming interface")
    
    print(f"\n💼 Interview talking points:")
    print(f"   - Built ML system that detects cognitive state from behavior")
    print(f"   - Achieved {best_accuracy:.1%} classification accuracy")
    print(f"   - Created novel behavioral simulation based on HCI research")
    print(f"   - End-to-end pipeline from data generation to real-time prediction")

if __name__ == "__main__":
    run_complete_week1_demo()
