#!/usr/bin/env python3
"""
MindState Demo - Basic Setup Test
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_setup():
    """Test that everything is set up correctly"""
    print("🧠 MindState Project Setup Test")
    print("=" * 40)
    
    # Test imports
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Core ML libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test data creation
    try:
        # Simple synthetic data
        data = {
            'timestamp': [datetime.now()] * 5,
            'typing_speed': np.random.normal(60, 10, 5),
            'mouse_precision': np.random.uniform(0.7, 0.95, 5),
            'cognitive_state': ['flow_state'] * 5
        }
        df = pd.DataFrame(data)
        print("✅ Synthetic data generation working")
        print(f"   Sample data shape: {df.shape}")
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False
    
    # Test directories
    directories = ['data', 'src', 'notebooks', 'tests']
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ Directory '{directory}' exists")
        else:
            print(f"❌ Directory '{directory}' missing")
    
    print("\n🎉 Setup test complete!")
    print("Ready to start building MindState!")
    return True

if __name__ == "__main__":
    test_setup()