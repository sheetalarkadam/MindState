"""
Simple test of LLM with behavioral data
"""

import sys
import os
sys.path.append('src/behavioral_analyzer')

def test_llm_with_behavioral_data():
    print("🧪 Testing LLM with Behavioral Data")
    print("=" * 40)
    
    try:
        from behavior_simulator import BehaviorDataGenerator
        import pandas as pd
        print("✅ Behavioral simulator imported")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return
    
    # Generate some test data
    generator = BehaviorDataGenerator()
    session = generator.generate_session_data('flow_state', duration_minutes=2)
    behavioral_df = pd.DataFrame(session['events'])
    
    # Create simple prompt
    avg_typing = behavioral_df['typing_speed'].mean()
    avg_errors = behavioral_df['typing_errors'].mean()
    avg_focus = behavioral_df['session_focus'].mean()
    
    prompt = f"""
    Analyze this behavioral data and determine cognitive state:
    
    - Typing speed: {avg_typing:.1f} WPM
    - Error rate: {avg_errors:.1%}  
    - Focus score: {avg_focus:.2f}
    
    The cognitive states are:
    - flow_state: high speed, low errors, high focus
    - stress_state: low speed, high errors, low focus
    - recovery_mode: slow speed, moderate errors, moderate focus
    
    Respond with JSON: {{"state": "state_name", "confidence": 0.XX}}
    """
    
    print("📤 Sending to LLM...")
    print(f"Actual state: flow_state")
    
    try:
        import ollama
        response = ollama.chat(
            model='llama2',
            messages=[
                {'role': 'system', 'content': 'You are a cognitive analyst. Respond only with JSON.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        result = response['message']['content']
        print(f"📥 LLM Response: {result}")
        
        # Try to parse JSON
        try:
            import json
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            json_str = result[json_start:json_end]
            parsed = json.loads(json_str)
            
            print(f"✅ Parsed successfully:")
            print(f"   Predicted: {parsed.get('state', 'unknown')}")
            print(f"   Confidence: {parsed.get('confidence', 0)}")
            
        except Exception as e:
            print(f"❌ JSON parsing failed: {e}")
            print("💡 LLM response wasn't valid JSON")
            
    except Exception as e:
        print(f"❌ LLM request failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")

# Run the test
if __name__ == "__main__":
    test_llm_with_behavioral_data()
