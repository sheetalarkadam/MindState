"""
LLM-Based Cognitive State Detection
Uses large language models for sophisticated behavioral pattern analysis
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os
import ollama

class LLMCognitiveAnalyzer:
    """Advanced cognitive state detection using Large Language Models"""
    
    def __init__(self):
        # Cognitive state definitions for LLM context
        self.cognitive_states_context = {
            "flow_state": {
                "description": "Deep focus with high productivity and minimal distractions",
                "behavioral_indicators": "fast accurate typing (70+ WPM), sustained attention (>0.85), minimal app switching (<0.2/min)",
                "optimal_content": "complex, challenging content requiring sustained attention"
            },
            "stress_state": {
                "description": "Overwhelmed state with high cognitive load and anxiety",
                "behavioral_indicators": "erratic typing (<50 WPM), many errors (>6%), frequent app switching (>1.5/min), poor focus (<0.4)",
                "optimal_content": "simple, calming content that reduces cognitive load"
            },
            "recovery_mode": {
                "description": "Low energy state needing mental rest and restoration",
                "behavioral_indicators": "slow deliberate actions (<40 WPM), long pauses, reduced activity, moderate focus (0.5-0.7)",
                "optimal_content": "familiar, easy-to-process content for relaxation"
            },
            "discovery_mode": {
                "description": "Curious and exploratory state with moderate engagement",
                "behavioral_indicators": "variable activity (50-70 WPM), exploratory clicking, moderate focus (0.6-0.8), some app switching",
                "optimal_content": "novel, moderately complex content for exploration"
            },
            "learning_mode": {
                "description": "Focused educational state with high motivation to absorb information",
                "behavioral_indicators": "deliberate typing (50-65 WPM), sustained focus (>0.8), careful navigation, few errors (<3%)",
                "optimal_content": "educational, substantial content with clear structure"
            }
        }
    
    def create_behavioral_prompt(self, behavioral_data: pd.DataFrame) -> str:
        """Create a detailed prompt for LLM analysis"""
        
        # Calculate comprehensive statistics
        stats = {
            'typing_speed_avg': behavioral_data['typing_speed'].mean(),
            'typing_speed_std': behavioral_data['typing_speed'].std(),
            'error_rate_avg': behavioral_data['typing_errors'].mean(),
            'focus_score_avg': behavioral_data['session_focus'].mean(),
            'focus_std': behavioral_data['session_focus'].std(),
            'app_switches_avg': behavioral_data['app_switches'].mean(),
            'mouse_precision_avg': behavioral_data['mouse_precision'].mean(),
            'session_length': len(behavioral_data)
        }
        
        # Analyze temporal patterns
        if len(behavioral_data) > 3:
            recent_focus = behavioral_data['session_focus'].tail(3).mean()
            early_focus = behavioral_data['session_focus'].head(3).mean()
            focus_trend = "improving" if recent_focus > early_focus else "declining"
            
            typing_trend = "increasing" if behavioral_data['typing_speed'].iloc[-1] > behavioral_data['typing_speed'].iloc[0] else "decreasing"
        else:
            focus_trend = "stable"
            typing_trend = "stable"
        
        prompt = f"""
You are an expert cognitive psychologist analyzing human-computer interaction patterns.

BEHAVIORAL DATA ANALYSIS:
- Typing Speed: {stats['typing_speed_avg']:.1f} WPM (variability: {stats['typing_speed_std']:.1f})
- Error Rate: {stats['error_rate_avg']:.1%}
- Focus Score: {stats['focus_score_avg']:.2f}/1.0 (variability: {stats['focus_std']:.2f})
- Mouse Precision: {stats['mouse_precision_avg']:.2f}/1.0
- App Switches: {stats['app_switches_avg']:.1f} per minute
- Focus Trend: {focus_trend} over session
- Typing Trend: {typing_trend} over session
- Data Points: {stats['session_length']} measurements

COGNITIVE STATES REFERENCE:
{json.dumps(self.cognitive_states_context, indent=2)}

ANALYSIS TASK:
1. Compare behavioral patterns to each cognitive state's indicators
2. Consider temporal trends and consistency
3. Assess confidence based on pattern strength
4. Identify supporting evidence

Respond with valid JSON only:
{{
    "predicted_state": "state_name",
    "confidence": 0.XX,
    "reasoning": "detailed explanation matching behavioral patterns to cognitive state",
    "behavioral_evidence": [
        "specific behavioral indicator supporting this classification",
        "another supporting behavioral pattern"
    ],
    "temporal_analysis": "analysis of how patterns changed over time",
    "alternative_states": {{
        "second_most_likely": "state_name",
        "probability": 0.XX,
        "why_considered": "brief explanation"
    }},
    "content_recommendations": "specific content types optimal for this cognitive state"
}}
"""
        return prompt
    
    async def analyze_with_llm(self, prompt: str) -> Dict:
        """Use LLM for cognitive analysis"""
        
        try:
            response = ollama.chat(
                model='llama2',
                messages=[
                    {
                        'role': 'system', 
                        'content': 'You are an expert cognitive psychologist. Respond only with valid JSON.'
                    },
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            result_text = response['message']['content']
            
            # Extract JSON from response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return self._fallback_analysis()
                
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> Dict:
        """Fallback analysis if LLM fails"""
        return {
            "predicted_state": "discovery_mode",
            "confidence": 0.6,
            "reasoning": "Fallback analysis due to LLM processing issue",
            "behavioral_evidence": ["moderate activity patterns detected"],
            "temporal_analysis": "insufficient data for temporal analysis",
            "alternative_states": {
                "second_most_likely": "learning_mode", 
                "probability": 0.3,
                "why_considered": "similar engagement patterns"
            },
            "content_recommendations": "moderate complexity content suitable for exploration"
        }
    
    async def predict_cognitive_state(self, behavioral_data: pd.DataFrame) -> Dict:
        """Main method to predict cognitive state using LLM"""
        
        if len(behavioral_data) < 3:
            return {
                "state": "insufficient_data",
                "confidence": 0.0,
                "reasoning": "Need at least 3 behavioral data points for reliable analysis",
                "method": "llm_analysis"
            }
        
        # Create detailed prompt
        prompt = self.create_behavioral_prompt(behavioral_data)
        
        # Analyze with LLM
        analysis = await self.analyze_with_llm(prompt)
        
        # Standardize output format
        return {
            "state": analysis.get("predicted_state", "unknown"),
            "confidence": analysis.get("confidence", 0.5),
            "reasoning": analysis.get("reasoning", "LLM cognitive analysis"),
            "behavioral_evidence": analysis.get("behavioral_evidence", []),
            "temporal_analysis": analysis.get("temporal_analysis", ""),
            "alternative_states": analysis.get("alternative_states", {}),
            "content_recommendations": analysis.get("content_recommendations", ""),
            "method": "llm_analysis",
            "timestamp": datetime.now().isoformat()
        }
    
    def explain_prediction(self, prediction: Dict, behavioral_data: pd.DataFrame) -> str:
        """Generate detailed natural language explanation"""
        
        state = prediction["state"].replace('_', ' ').title()
        confidence = prediction["confidence"]
        reasoning = prediction.get("reasoning", "")
        
        explanation = f"""
🧠 **Cognitive State Analysis: {state}**

**Confidence Level:** {confidence:.1%}

**Analysis:**
{reasoning}

**Supporting Evidence:**
"""
        
        for evidence in prediction.get("behavioral_evidence", []):
            explanation += f"• {evidence}\n"
        
        if prediction.get("temporal_analysis"):
            explanation += f"""
**Temporal Patterns:**
{prediction['temporal_analysis']}
"""
        
        explanation += f"""
**Content Recommendations:**
{prediction.get('content_recommendations', 'Content matched to current cognitive capacity')}

**Alternative Consideration:**
{prediction.get('alternative_states', {}).get('second_most_likely', 'N/A')} (probability: {prediction.get('alternative_states', {}).get('probability', 0):.1%})

**Session Quality:** Analyzed {len(behavioral_data)} behavioral data points
"""
        
        return explanation

def demo_llm_cognitive_analysis():
    """Demonstrate advanced LLM-based cognitive analysis"""
    
    print("�� Advanced LLM-Based Cognitive State Analysis Demo")
    print("=" * 60)
    
    # Import behavioral data generator
    sys.path.append(os.path.dirname(__file__))
    from behavior_simulator import BehaviorDataGenerator
    
    # Generate sample behavioral data
    generator = BehaviorDataGenerator()
    analyzer = LLMCognitiveAnalyzer()
    
    print("\n1. Testing LLM analysis on different cognitive states...")
    
    # Test different cognitive states
    test_states = ['flow_state', 'stress_state', 'learning_mode', 'recovery_mode']
    
    for state in test_states:
        print(f"\n🧪 Testing {state}:")
        print("-" * 35)
        
        # Generate realistic behavioral session
        session = generator.generate_session_data(state, duration_minutes=4)
        behavioral_df = pd.DataFrame(session['events'])
        
        # Analyze with LLM
        import asyncio
        analysis = asyncio.run(analyzer.predict_cognitive_state(behavioral_df))
        
        print(f"🎯 LLM Prediction: {analysis['state']} (confidence: {analysis['confidence']:.1%})")
        print(f"✅ Actual State: {state}")
        print(f"📝 Reasoning: {analysis['reasoning'][:120]}...")
        
        # Show evidence
        if analysis.get('behavioral_evidence'):
            print(f"📊 Key Evidence:")
            for evidence in analysis['behavioral_evidence'][:2]:
                print(f"   • {evidence}")
        
        # Show alternative
        alt_states = analysis.get('alternative_states', {})
        if alt_states.get('second_most_likely'):
            print(f"🔄 Alternative: {alt_states['second_most_likely']} ({alt_states.get('probability', 0):.1%})")
        
        is_correct = analysis['state'] == state
        print(f"\n{'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        if not is_correct:
            print(f"💡 LLM reasoning: {analysis['reasoning'][:200]}...")
    
    print(f"\n🎉 LLM Cognitive Analysis Demo Complete!")
    print(f"🚀 Advantages over traditional ML:")
    print(f"   • Rich contextual reasoning and explanations")
    print(f"   • Understanding of behavioral nuances")
    print(f"   • Temporal pattern analysis")
    print(f"   • Alternative state consideration")
    print(f"   • Natural language content recommendations")

if __name__ == "__main__":
    demo_llm_cognitive_analysis()
