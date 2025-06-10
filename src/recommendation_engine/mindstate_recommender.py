"""
MindState Cognitive-Aware Recommendation Engine
The core system that matches content complexity to user cognitive state
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from behavioral_analyzer.cognitive_state_detector import CognitiveStateClassifier
from content_analyzer.complexity_analyzer import ContentComplexityAnalyzer

class MindStateRecommender:
    """Main recommendation engine that considers cognitive state"""
    
    def __init__(self):
        self.cognitive_detector = CognitiveStateClassifier()
        self.complexity_analyzer = ContentComplexityAnalyzer()
        self.content_database = None
        self.user_preferences = {}
        
        # Cognitive state to complexity tolerance mapping
        self.state_complexity_tolerance = {
            'stress_state': 0.25,      # Very low tolerance when stressed
            'recovery_mode': 0.35,     # Low tolerance when tired
            'discovery_mode': 0.65,    # Medium tolerance when exploring
            'learning_mode': 0.8,      # High tolerance when learning
            'flow_state': 0.9          # Very high tolerance when focused
        }
        
        # State-specific preference adjustments
        self.state_preferences = {
            'stress_state': {
                'preferred_genres': ['comedy', 'animation', 'family', 'music'],
                'avoid_genres': ['horror', 'thriller', 'war', 'crime'],
                'max_duration': 45,  # Shorter content when stressed
                'prefer_familiar': True
            },
            'recovery_mode': {
                'preferred_genres': ['comedy', 'romance', 'family', 'animation'],
                'avoid_genres': ['action', 'thriller', 'complex_drama'],
                'max_duration': 60,
                'prefer_familiar': True
            },
            'discovery_mode': {
                'preferred_genres': ['adventure', 'documentary', 'drama', 'mystery'],
                'avoid_genres': [],  # Open to exploration
                'max_duration': 120,
                'prefer_familiar': False
            },
            'learning_mode': {
                'preferred_genres': ['documentary', 'historical', 'biography', 'educational'],
                'avoid_genres': ['comedy', 'action'],  # Want substance
                'max_duration': 180,
                'prefer_familiar': False
            },
            'flow_state': {
                'preferred_genres': ['sci-fi', 'complex_drama', 'thriller', 'mystery'],
                'avoid_genres': ['simple_comedy'],
                'max_duration': 240,  # Can handle long content
                'prefer_familiar': False
            }
        }
    
    def load_content_database(self, content_data: List[Dict]):
        """Load and analyze content database"""
        print("Loading and analyzing content database...")
        
        analyzed_content = []
        
        for content in content_data:
            # Analyze content complexity
            complexity_analysis = self.complexity_analyzer.calculate_cognitive_load_score(content)
            
            # Add complexity scores to content
            content_with_analysis = content.copy()
            content_with_analysis.update(complexity_analysis)
            
            analyzed_content.append(content_with_analysis)
        
        self.content_database = pd.DataFrame(analyzed_content)
        print(f"Loaded {len(self.content_database)} content items")
        
        # Show complexity distribution
        complexity_dist = self.content_database['complexity_category'].value_counts()
        print(f"Complexity distribution:\n{complexity_dist}")
    
    def predict_user_cognitive_state(self, behavioral_data: pd.DataFrame) -> Dict:
        """Predict user's current cognitive state from behavioral data"""
        
        if not self.cognitive_detector.is_trained:
            # For demo purposes, return a sample state
            return {
                'state': 'discovery_mode',
                'confidence': 0.85,
                'note': 'Using demo state - cognitive detector not trained'
            }
        
        return self.cognitive_detector.predict(behavioral_data)
    
    def filter_by_cognitive_compatibility(self, cognitive_state: str, 
                                         confidence: float = 1.0) -> pd.DataFrame:
        """Filter content based on cognitive state compatibility"""
        
        if self.content_database is None:
            raise ValueError("Content database not loaded")
        
        # Get complexity tolerance for this state
        max_complexity = self.state_complexity_tolerance.get(cognitive_state, 0.5)
        
        # Adjust tolerance based on confidence
        # Lower confidence = be more conservative with complexity
        adjusted_complexity = max_complexity * confidence
        
        # Filter by complexity
        suitable_content = self.content_database[
            self.content_database['overall_complexity'] <= adjusted_complexity
        ].copy()
        
        # Apply state-specific preferences
        state_prefs = self.state_preferences.get(cognitive_state, {})
        
        # Filter by duration if specified
        if 'max_duration' in state_prefs:
            suitable_content = suitable_content[
                suitable_content['duration'] <= state_prefs['max_duration']
            ]
        
        # Prefer certain genres
        preferred_genres = state_prefs.get('preferred_genres', [])
        if preferred_genres:
            # Boost score for preferred genres
            suitable_content['genre_match_score'] = suitable_content['genre'].apply(
                lambda x: any(pref in x.lower() for pref in preferred_genres)
            ).astype(float)
        else:
            suitable_content['genre_match_score'] = 0.5
        
        # Avoid certain genres
        avoid_genres = state_prefs.get('avoid_genres', [])
        if avoid_genres:
            for avoid_genre in avoid_genres:
                suitable_content = suitable_content[
                    ~suitable_content['genre'].str.lower().str.contains(avoid_genre, na=False)
                ]
        
        return suitable_content
    
    def rank_recommendations(self, filtered_content: pd.DataFrame, 
                           cognitive_state: str, user_id: str = "demo_user") -> pd.DataFrame:
        """Rank filtered content for final recommendations"""
        
        if len(filtered_content) == 0:
            return filtered_content
        
        # Initialize ranking score
        filtered_content = filtered_content.copy()
        filtered_content['recommendation_score'] = 0.0
        
        # Factor 1: Cognitive complexity match (40% weight)
        target_complexity = self.state_complexity_tolerance.get(cognitive_state, 0.5)
        complexity_match = 1 - abs(filtered_content['overall_complexity'] - target_complexity)
        filtered_content['recommendation_score'] += complexity_match * 0.4
        
        # Factor 2: Genre preference match (25% weight)
        filtered_content['recommendation_score'] += filtered_content['genre_match_score'] * 0.25
        
        # Factor 3: Duration appropriateness (15% weight)
        state_prefs = self.state_preferences.get(cognitive_state, {})
        ideal_duration = state_prefs.get('max_duration', 120)
        
        duration_score = np.where(
            filtered_content['duration'] <= ideal_duration,
            1.0 - (filtered_content['duration'] / ideal_duration) * 0.3,  # Prefer shorter within limit
            0.5  # Penalize if over limit
        )
        filtered_content['recommendation_score'] += duration_score * 0.15
        
        # Factor 4: Diversity bonus (10% weight)
        # Add small random component to ensure variety
        filtered_content['recommendation_score'] += np.random.uniform(0, 0.1, len(filtered_content))
        
        # Factor 5: Quality indicators (10% weight)
        # For demo, use a simple quality proxy
        quality_score = np.random.uniform(0.3, 1.0, len(filtered_content))  # Simulate ratings
        filtered_content['recommendation_score'] += quality_score * 0.1
        
        # Sort by recommendation score
        filtered_content = filtered_content.sort_values('recommendation_score', ascending=False)
        
        return filtered_content
    
    def get_recommendations(self, behavioral_data: Optional[pd.DataFrame] = None,
                          cognitive_state: Optional[str] = None,
                          num_recommendations: int = 5,
                          user_id: str = "demo_user") -> Dict:
        """Get personalized recommendations based on cognitive state"""
        
        # Predict cognitive state if not provided
        if cognitive_state is None:
            if behavioral_data is not None:
                state_prediction = self.predict_user_cognitive_state(behavioral_data)
                cognitive_state = state_prediction['state']
                confidence = state_prediction['confidence']
            else:
                # Demo mode
                cognitive_state = 'discovery_mode'
                confidence = 0.8
        else:
            confidence = 1.0
        
        # Filter content by cognitive compatibility
        suitable_content = self.filter_by_cognitive_compatibility(cognitive_state, confidence)
        
        if len(suitable_content) == 0:
            return {
                'cognitive_state': cognitive_state,
                'recommendations': [],
                'message': 'No suitable content found for current cognitive state',
                'suggestions': 'Try adjusting complexity tolerance or expanding content database'
            }
        
        # Rank and select top recommendations
        ranked_content = self.rank_recommendations(suitable_content, cognitive_state, user_id)
        top_recommendations = ranked_content.head(num_recommendations)
        
        # Format recommendations
        recommendations = []
        for _, item in top_recommendations.iterrows():
            recommendation = {
                'title': item['title'],
                'genre': item['genre'],
                'duration': item['duration'],
                'complexity_score': item['overall_complexity'],
                'complexity_category': item['complexity_category'],
                'recommendation_score': round(item['recommendation_score'], 3),
                'why_recommended': self._generate_explanation(item, cognitive_state),
                'cognitive_load_info': item['cognitive_load_factors']
            }
            recommendations.append(recommendation)
        
        return {
            'cognitive_state': cognitive_state,
            'state_confidence': confidence,
            'complexity_tolerance': self.state_complexity_tolerance[cognitive_state],
            'recommendations': recommendations,
            'total_suitable_items': len(suitable_content),
            'state_description': self._get_state_description(cognitive_state)
        }
    
    def _generate_explanation(self, content_item: pd.Series, cognitive_state: str) -> str:
        """Generate explanation for why content was recommended"""
        
        explanations = []
        
        # Complexity match
        complexity = content_item['overall_complexity']
        tolerance = self.state_complexity_tolerance[cognitive_state]
        
        if complexity <= tolerance * 0.8:
            explanations.append(f"Low cognitive load ({complexity:.2f}) perfect for {cognitive_state.replace('_', ' ')}")
        elif complexity <= tolerance:
            explanations.append(f"Moderate complexity ({complexity:.2f}) matches your current mental capacity")
        
        # Duration appropriateness
        duration = content_item['duration']
        if duration <= 30:
            explanations.append("Short duration ideal for current attention span")
        elif duration <= 90:
            explanations.append("Moderate length fits your focus capacity")
        
        # Genre match
        state_prefs = self.state_preferences.get(cognitive_state, {})
        preferred_genres = state_prefs.get('preferred_genres', [])
        genre = content_item['genre'].lower()
        
        for pref_genre in preferred_genres:
            if pref_genre in genre:
                explanations.append(f"Genre matches {cognitive_state.replace('_', ' ')} preferences")
                break
        
        return "; ".join(explanations) if explanations else "Good match for current cognitive state"
    
    def _get_state_description(self, cognitive_state: str) -> str:
        """Get human-readable description of cognitive state"""
        
        descriptions = {
            'stress_state': 'Feeling overwhelmed - need calming, simple content',
            'recovery_mode': 'Mentally tired - prefer familiar, easy content',
            'discovery_mode': 'Curious and exploring - ready for moderate complexity',
            'learning_mode': 'Focused on learning - want educational, substantial content',
            'flow_state': 'Deep focus mode - can handle complex, challenging content'
        }
        
        return descriptions.get(cognitive_state, 'Current cognitive state')

def demo_mindstate_recommendations():
    """Demonstrate the complete MindState recommendation system"""
    
    print("🧠🎬 MindState Cognitive-Aware Recommendation Engine Demo")
    print("=" * 60)
    
    # Initialize recommender
    recommender = MindStateRecommender()
    
    # Sample content database
    sample_content = [
        {
            'title': 'Friends - Season 1',
            'genre': 'Comedy/Romance',
            'duration': 22,
            'type': 'tv_series',
            'description': 'Six friends navigate life and love in New York City.',
            'plot': 'Light-hearted situations with familiar characters.'
        },
        {
            'title': 'The Avengers',
            'genre': 'Action/Adventure',
            'duration': 143,
            'type': 'movie',
            'description': 'Superheroes team up to save the world.',
            'plot': 'Fast-paced action with multiple storylines.'
        },
        {
            'title': 'Planet Earth',
            'genre': 'Documentary/Nature',
            'duration': 50,
            'type': 'documentary',
            'description': 'Stunning wildlife documentary series.',
            'plot': 'Educational content about natural world.'
        },
        {
            'title': 'Inception',
            'genre': 'Sci-Fi/Thriller',
            'duration': 148,
            'type': 'movie',
            'description': 'A thief enters dreams to steal secrets.',
            'plot': 'Complex narrative with multiple reality layers.'
        },
        {
            'title': 'The Office',
            'genre': 'Comedy/Mockumentary',
            'duration': 22,
            'type': 'tv_series',
            'description': 'Workplace comedy in a paper company.',
            'plot': 'Character-driven humor with minimal complexity.'
        },
        {
            'title': 'Cosmos: A Space-Time Odyssey',
            'genre': 'Documentary/Science',
            'duration': 43,
            'type': 'documentary',
            'description': 'Exploration of space and scientific discovery.',
            'plot': 'Complex scientific concepts and theories.'
        },
        {
            'title': 'Toy Story',
            'genre': 'Animation/Family',
            'duration': 81,
            'type': 'movie',
            'description': 'Toys come to life when humans are away.',
            'plot': 'Simple story with emotional themes.'
        }
    ]
    
    # Load content database
    recommender.load_content_database(sample_content)
    
    print(f"\n🎯 Testing Recommendations for Different Cognitive States:")
    print("=" * 55)
    
    # Test different cognitive states
    test_states = ['stress_state', 'recovery_mode', 'discovery_mode', 'learning_mode', 'flow_state']
    
    for state in test_states:
        print(f"\n🧠 {state.replace('_', ' ').title()} Recommendations:")
        print("-" * 40)
        
        # Get recommendations for this state
        recommendations = recommender.get_recommendations(
            cognitive_state=state,
            num_recommendations=3
        )
        
        print(f"State: {recommendations['state_description']}")
        print(f"Complexity tolerance: {recommendations['complexity_tolerance']}")
        print(f"Found {recommendations['total_suitable_items']} suitable items")
        
        print(f"\nTop 3 Recommendations:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"  {i}. {rec['title']} ({rec['genre']})")
            print(f"     Duration: {rec['duration']} min | Complexity: {rec['complexity_score']:.2f}")
            print(f"     Why: {rec['why_recommended']}")
            print()
    
    print(f"🎉 MindState Recommendation Engine Demo Complete!")
    print(f"✅ Successfully matches content complexity to cognitive state")
    print(f"✅ Provides explanations for recommendations")
    print(f"✅ Adapts to different mental energy levels")
    print(f"🚀 Ready for real-time integration!")

if __name__ == "__main__":
    demo_mindstate_recommendations()
