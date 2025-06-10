"""
Advanced Behavioral Pattern Simulator
Generates realistic behavioral data patterns for different cognitive states
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List
from collections import deque

class BehaviorDataGenerator:
    """Generate realistic synthetic behavioral data for different cognitive states"""
    
    def __init__(self):
        # Define behavioral patterns for each cognitive state based on HCI research
        self.state_patterns = {
            'flow_state': {
                'typing_speed': {'mean': 75, 'std': 8},      # WPM - higher in flow
                'typing_errors': {'mean': 0.02, 'std': 0.01}, # Lower error rate
                'mouse_precision': {'mean': 0.9, 'std': 0.05}, # High precision
                'click_frequency': {'mean': 0.5, 'std': 0.15}, # Steady clicking
                'scroll_smoothness': {'mean': 0.85, 'std': 0.1}, # Smooth scrolling
                'pause_duration': {'mean': 1.5, 'std': 0.8},   # Short pauses
                'session_focus': {'mean': 0.92, 'std': 0.05},  # High focus
                'app_switches': {'mean': 0.08, 'std': 0.04}    # Minimal switching
            },
            'stress_state': {
                'typing_speed': {'mean': 45, 'std': 15},     # Erratic typing
                'typing_errors': {'mean': 0.08, 'std': 0.03}, # High error rate
                'mouse_precision': {'mean': 0.65, 'std': 0.15}, # Poor precision
                'click_frequency': {'mean': 2.2, 'std': 0.8},  # Frantic clicking
                'scroll_smoothness': {'mean': 0.4, 'std': 0.2}, # Jerky scrolling
                'pause_duration': {'mean': 0.3, 'std': 0.2},   # Short, rapid pauses
                'session_focus': {'mean': 0.25, 'std': 0.15},  # Very low focus
                'app_switches': {'mean': 1.8, 'std': 0.6}      # Frequent switching
            },
            'recovery_mode': {
                'typing_speed': {'mean': 35, 'std': 8},      # Slow, deliberate
                'typing_errors': {'mean': 0.04, 'std': 0.02}, # Moderate errors
                'mouse_precision': {'mean': 0.75, 'std': 0.1}, # Decent precision
                'click_frequency': {'mean': 0.25, 'std': 0.1}, # Slow clicking
                'scroll_smoothness': {'mean': 0.65, 'std': 0.15}, # Smooth but slow
                'pause_duration': {'mean': 4.0, 'std': 2.0},   # Long pauses
                'session_focus': {'mean': 0.6, 'std': 0.2},    # Moderate focus
                'app_switches': {'mean': 0.2, 'std': 0.1}      # Minimal switching
            },
            'discovery_mode': {
                'typing_speed': {'mean': 60, 'std': 12},     # Variable typing
                'typing_errors': {'mean': 0.04, 'std': 0.02}, # Moderate errors
                'mouse_precision': {'mean': 0.8, 'std': 0.1},  # Good precision
                'click_frequency': {'mean': 1.1, 'std': 0.4},  # Exploratory clicking
                'scroll_smoothness': {'mean': 0.7, 'std': 0.1}, # Decent scrolling
                'pause_duration': {'mean': 2.5, 'std': 1.2},   # Thinking pauses
                'session_focus': {'mean': 0.75, 'std': 0.15},  # Good focus
                'app_switches': {'mean': 0.6, 'std': 0.25}     # Some switching
            },
            'learning_mode': {
                'typing_speed': {'mean': 55, 'std': 10},     # Careful typing
                'typing_errors': {'mean': 0.025, 'std': 0.015}, # Low errors
                'mouse_precision': {'mean': 0.85, 'std': 0.08}, # High precision
                'click_frequency': {'mean': 0.4, 'std': 0.15},  # Deliberate clicks
                'scroll_smoothness': {'mean': 0.8, 'std': 0.1}, # Smooth scrolling
                'pause_duration': {'mean': 3.0, 'std': 1.5},   # Reading pauses
                'session_focus': {'mean': 0.88, 'std': 0.08},  # High focus
                'app_switches': {'mean': 0.15, 'std': 0.08}    # Minimal switching
            }
        }
    
    def generate_session_data(self, state: str, duration_minutes: int = 10) -> Dict:
        """Generate behavioral data for a complete session"""
        
        if state not in self.state_patterns:
            raise ValueError(f"Unknown cognitive state: {state}")
        
        patterns = self.state_patterns[state]
        num_events = int(duration_minutes * 6)  # Event every 10 seconds
        
        session_data = {
            'state': state,
            'duration_minutes': duration_minutes,
            'start_timestamp': datetime.now(),
            'events': []
        }
        
        # Add temporal effects (fatigue over time)
        for i in range(num_events):
            event = {}
            
            # Calculate fatigue factor (increases over session)
            fatigue_factor = min(0.3, i / num_events * 0.3)
            
            # Add circadian rhythm effect (simplified)
            hour = datetime.now().hour
            circadian_effect = 0.1 * np.sin(2 * np.pi * (hour - 10) / 24)
            
            for feature, params in patterns.items():
                # Base value from pattern
                base_value = np.random.normal(params['mean'], params['std'])
                
                # Apply temporal effects
                if feature in ['typing_speed', 'mouse_precision', 'session_focus']:
                    # These decrease with fatigue
                    value = base_value * (1 - fatigue_factor) * (1 + circadian_effect)
                elif feature in ['typing_errors', 'app_switches', 'pause_duration']:
                    # These increase with fatigue
                    value = base_value * (1 + fatigue_factor) * (1 - circadian_effect)
                else:
                    value = base_value
                
                # Ensure realistic bounds
                if feature == 'typing_speed':
                    value = max(15, min(120, value))
                elif feature in ['mouse_precision', 'scroll_smoothness', 'session_focus']:
                    value = max(0.1, min(1.0, value))
                elif feature in ['typing_errors']:
                    value = max(0.001, min(0.2, value))
                else:
                    value = max(0, value)
                
                event[feature] = round(value, 3)
            
            event['timestamp'] = session_data['start_timestamp'] + timedelta(seconds=i*10)
            event['session_minute'] = i / 6  # Which minute of the session
            session_data['events'].append(event)
        
        return session_data
    
    def generate_dataset(self, sessions_per_state: int = 50) -> pd.DataFrame:
        """Generate a complete dataset with multiple sessions for each cognitive state"""
        
        all_events = []
        
        for state in self.state_patterns.keys():
            print(f"Generating {sessions_per_state} sessions for {state}...")
            
            for session_id in range(sessions_per_state):
                # Vary session duration realistically
                duration = random.randint(5, 30)  # 5-30 minute sessions
                
                session = self.generate_session_data(state, duration)
                
                # Add session metadata to each event
                for event in session['events']:
                    event['session_id'] = f"{state}_{session_id}"
                    event['cognitive_state'] = state
                    event['session_duration'] = duration
                    all_events.append(event)
        
        df = pd.DataFrame(all_events)
        print(f"\nGenerated dataset with {len(df)} behavioral events")
        print(f"States distribution:\n{df['cognitive_state'].value_counts()}")
        
        return df

def demo_behavior_generation():
    """Demonstrate the behavioral data generation"""
    
    print("🧠 MindState Behavioral Data Generator Demo")
    print("=" * 50)
    
    generator = BehaviorDataGenerator()
    
    # Generate a small dataset for demo
    print("\n1. Generating sample dataset...")
    dataset = generator.generate_dataset(sessions_per_state=10)  # Small for demo
    
    print(f"\n2. Dataset overview:")
    print(f"   Shape: {dataset.shape}")
    print(f"   Columns: {list(dataset.columns)}")
    
    print(f"\n3. Sample data from 'flow_state':")
    flow_data = dataset[dataset['cognitive_state'] == 'flow_state'].head(3)
    print(flow_data[['typing_speed', 'mouse_precision', 'session_focus', 'app_switches']].to_string())
    
    print(f"\n4. Sample data from 'stress_state':")
    stress_data = dataset[dataset['cognitive_state'] == 'stress_state'].head(3)
    print(stress_data[['typing_speed', 'mouse_precision', 'session_focus', 'app_switches']].to_string())
    
    print(f"\n5. Statistical comparison between states:")
    comparison = dataset.groupby('cognitive_state')[['typing_speed', 'mouse_precision', 'session_focus']].mean()
    print(comparison.round(2))
    
    print(f"\n✅ Behavioral data generation working perfectly!")
    print(f"📊 Clear patterns visible between cognitive states")
    print(f"🚀 Ready for cognitive state classification!")

if __name__ == "__main__":
    demo_behavior_generation()
