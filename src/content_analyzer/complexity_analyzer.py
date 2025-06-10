"""
Content Complexity Analyzer
Analyzes and scores content based on cognitive load requirements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade, syllable_count
from collections import Counter
import json

class ContentComplexityAnalyzer:
    """Analyzes content complexity across multiple dimensions"""
    
    def __init__(self):
        # Genre complexity mappings based on cognitive load research
        self.genre_complexity = {
            # Low complexity genres
            'comedy': 0.2, 'animation': 0.25, 'family': 0.2, 'romance': 0.3,
            'music': 0.15, 'documentary': 0.4,
            
            # Medium complexity genres  
            'action': 0.5, 'adventure': 0.55, 'drama': 0.6, 'thriller': 0.65,
            'crime': 0.7, 'mystery': 0.7, 'horror': 0.6,
            
            # High complexity genres
            'sci-fi': 0.85, 'fantasy': 0.8, 'war': 0.75, 'historical': 0.8,
            'biography': 0.7, 'political': 0.9, 'philosophical': 0.95
        }
        
        # Content type complexity multipliers
        self.content_type_multipliers = {
            'movie': 1.0,
            'tv_series': 1.2,  # More complex due to ongoing plots
            'documentary': 1.3,  # Educational content requires more focus
            'tutorial': 1.4,  # Learning content is cognitively demanding
            'news': 0.8,  # Usually bite-sized information
            'social_media': 0.3  # Designed for easy consumption
        }
    
    def analyze_text_complexity(self, text: str, title: str = "") -> Dict:
        """Analyze text-based content complexity"""
        
        if not text or len(text.strip()) < 10:
            return {'text_complexity': 0.5, 'readability_score': 0.5, 'concept_density': 0.5}
        
        # Basic text metrics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Readability scores (lower = easier to read)
        try:
            flesch_score = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            # Convert to 0-1 scale where higher = more complex
            readability_complexity = max(0, min(1, (100 - flesch_score) / 100))
        except:
            readability_complexity = 0.5  # Default if calculation fails
        
        # Vocabulary complexity
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / max(len(words), 1)
        
        # Complex word analysis (words with 3+ syllables)
        complex_words = 0
        for word in unique_words:
            try:
                if syllable_count(word) >= 3:
                    complex_words += 1
            except:
                pass
        
        complex_word_ratio = complex_words / max(len(unique_words), 1)
        
        # Concept density (technical terms, proper nouns)
        technical_indicators = [
            'analysis', 'system', 'process', 'method', 'theory', 'research',
            'data', 'algorithm', 'framework', 'implementation', 'optimization'
        ]
        
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        technical_terms = sum(1 for indicator in technical_indicators if indicator in text.lower())
        
        concept_density = min(1.0, (proper_nouns + technical_terms) / max(word_count / 100, 1))
        
        # Overall text complexity
        text_complexity = np.mean([
            readability_complexity * 0.4,
            vocabulary_diversity * 0.3,
            complex_word_ratio * 0.2,
            concept_density * 0.1
        ])
        
        return {
            'text_complexity': round(text_complexity, 3),
            'readability_score': round(readability_complexity, 3),
            'vocabulary_diversity': round(vocabulary_diversity, 3),
            'complex_word_ratio': round(complex_word_ratio, 3),
            'concept_density': round(concept_density, 3),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'word_count': word_count
        }
    
    def analyze_media_complexity(self, duration_minutes: float, genre: str, 
                                content_type: str = 'movie') -> Dict:
        """Analyze media content complexity based on duration, genre, type"""
        
        # Duration complexity (longer content requires more sustained attention)
        if duration_minutes <= 30:
            duration_complexity = 0.2  # Short content
        elif duration_minutes <= 90:
            duration_complexity = 0.5  # Medium content
        elif duration_minutes <= 180:
            duration_complexity = 0.8  # Long content
        else:
            duration_complexity = 1.0  # Very long content
        
        # Genre complexity
        genre_lower = genre.lower()
        genre_complexity = 0.5  # Default
        
        for genre_key, complexity in self.genre_complexity.items():
            if genre_key in genre_lower:
                genre_complexity = complexity
                break
        
        # Content type multiplier
        type_multiplier = self.content_type_multipliers.get(content_type.lower(), 1.0)
        
        # Combined media complexity
        media_complexity = min(1.0, (duration_complexity * 0.4 + genre_complexity * 0.6) * type_multiplier)
        
        return {
            'media_complexity': round(media_complexity, 3),
            'duration_complexity': round(duration_complexity, 3),
            'genre_complexity': round(genre_complexity, 3),
            'content_type_multiplier': round(type_multiplier, 2),
            'duration_minutes': duration_minutes
        }
    
    def calculate_cognitive_load_score(self, content_data: Dict) -> Dict:
        """Calculate overall cognitive load score for content"""
        
        # Initialize components
        text_score = 0.5
        media_score = 0.5
        interaction_score = 0.0
        
        # Text complexity component
        if 'description' in content_data or 'plot' in content_data:
            text_content = content_data.get('description', '') + ' ' + content_data.get('plot', '')
            title = content_data.get('title', '')
            text_analysis = self.analyze_text_complexity(text_content, title)
            text_score = text_analysis['text_complexity']
        
        # Media complexity component
        if 'duration' in content_data and 'genre' in content_data:
            duration = content_data['duration']
            genre = content_data['genre']
            content_type = content_data.get('type', 'movie')
            
            media_analysis = self.analyze_media_complexity(duration, genre, content_type)
            media_score = media_analysis['media_complexity']
        
        # Interaction complexity (for interactive content)
        if content_data.get('interactive', False):
            interaction_score = 0.3  # Base interaction complexity
            
            # Add complexity for decision points
            decision_points = content_data.get('decision_points', 0)
            interaction_score += min(0.4, decision_points * 0.1)
            
            # Add complexity for multitasking requirements
            if content_data.get('requires_multitasking', False):
                interaction_score += 0.3
        
        # Weighted combination
        weights = {
            'text': 0.3,
            'media': 0.5, 
            'interaction': 0.2
        }
        
        overall_complexity = (
            text_score * weights['text'] +
            media_score * weights['media'] +
            interaction_score * weights['interaction']
        )
        
        # Ensure score is between 0 and 1
        overall_complexity = max(0.0, min(1.0, overall_complexity))
        
        # Complexity categories
        if overall_complexity <= 0.3:
            complexity_category = 'Low'
            recommendation = 'Good for tired/stressed states'
        elif overall_complexity <= 0.6:
            complexity_category = 'Medium'  
            recommendation = 'Good for moderate energy states'
        else:
            complexity_category = 'High'
            recommendation = 'Best for high energy/flow states'
        
        return {
            'overall_complexity': round(overall_complexity, 3),
            'complexity_category': complexity_category,
            'recommendation': recommendation,
            'component_scores': {
                'text_complexity': round(text_score, 3),
                'media_complexity': round(media_score, 3), 
                'interaction_complexity': round(interaction_score, 3)
            },
            'cognitive_load_factors': {
                'attention_span_required': self._estimate_attention_span(overall_complexity),
                'working_memory_load': self._estimate_memory_load(text_score, interaction_score),
                'processing_speed_required': self._estimate_processing_speed(media_score)
            }
        }
    
    def _estimate_attention_span(self, complexity: float) -> str:
        """Estimate required attention span based on complexity"""
        if complexity <= 0.3:
            return "5-15 minutes"
        elif complexity <= 0.6:
            return "15-45 minutes"
        else:
            return "45+ minutes"
    
    def _estimate_memory_load(self, text_complexity: float, interaction_complexity: float) -> str:
        """Estimate working memory requirements"""
        memory_load = (text_complexity + interaction_complexity) / 2
        
        if memory_load <= 0.3:
            return "Low - minimal information to track"
        elif memory_load <= 0.6:
            return "Medium - moderate information tracking"
        else:
            return "High - complex information management"
    
    def _estimate_processing_speed(self, media_complexity: float) -> str:
        """Estimate processing speed requirements"""
        if media_complexity <= 0.3:
            return "Relaxed - plenty of time to process"
        elif media_complexity <= 0.6:
            return "Moderate - normal processing pace"
        else:
            return "Fast - rapid information processing needed"

def demo_content_complexity():
    """Demonstrate content complexity analysis"""
    
    print("🎬 MindState Content Complexity Analyzer Demo")
    print("=" * 50)
    
    analyzer = ContentComplexityAnalyzer()
    
    # Sample content for analysis
    sample_content = [
        {
            'title': 'The Avengers',
            'genre': 'Action/Adventure',
            'duration': 143,
            'type': 'movie',
            'description': 'A team of superheroes comes together to save the world from an alien invasion.',
            'plot': 'Fast-paced action sequences with multiple storylines and characters.'
        },
        {
            'title': 'Quantum Physics Explained',
            'genre': 'Documentary/Educational',
            'duration': 87,
            'type': 'documentary',
            'description': 'An in-depth exploration of quantum mechanics, wave-particle duality, and quantum entanglement.',
            'plot': 'Complex scientific concepts explained through mathematical frameworks and experimental evidence.'
        },
        {
            'title': 'Friends',
            'genre': 'Comedy/Romance',
            'duration': 22,
            'type': 'tv_series',
            'description': 'A group of friends navigate life and relationships in New York City.',
            'plot': 'Light-hearted situations with familiar characters and predictable story arcs.'
        },
        {
            'title': 'Inception',
            'genre': 'Sci-Fi/Thriller',
            'duration': 148,
            'type': 'movie',
            'description': 'A thief who steals corporate secrets through dream-sharing technology.',
            'plot': 'Multiple layers of reality, complex narrative structure, philosophical themes about consciousness and reality.'
        }
    ]
    
    print("Analyzing content complexity...\n")
    
    for i, content in enumerate(sample_content, 1):
        print(f"{i}. {content['title']} ({content['genre']})")
        print(f"   Duration: {content['duration']} minutes")
        
        # Analyze complexity
        analysis = analyzer.calculate_cognitive_load_score(content)
        
        print(f"   📊 Complexity Score: {analysis['overall_complexity']:.2f} ({analysis['complexity_category']})")
        print(f"   🎯 {analysis['recommendation']}")
        print(f"   ⏱️  Attention span needed: {analysis['cognitive_load_factors']['attention_span_required']}")
        print(f"   🧠 Memory load: {analysis['cognitive_load_factors']['working_memory_load']}")
        print()
    
    print("🧠 Cognitive State Matching Examples:")
    print("=" * 40)
    
    cognitive_states = {
        'stress_state': 0.2,      # Need low complexity
        'recovery_mode': 0.3,     # Need easy content
        'discovery_mode': 0.6,    # Can handle medium complexity
        'learning_mode': 0.8,     # Can handle high complexity
        'flow_state': 0.9         # Can handle very high complexity
    }
    
    for state, tolerance in cognitive_states.items():
        print(f"\n{state.replace('_', ' ').title()} (tolerance: {tolerance}):")
        suitable_content = []
        
        for content in sample_content:
            analysis = analyzer.calculate_cognitive_load_score(content)
            if analysis['overall_complexity'] <= tolerance:
                suitable_content.append(f"  ✅ {content['title']} ({analysis['overall_complexity']:.2f})")
            else:
                suitable_content.append(f"  ❌ {content['title']} ({analysis['overall_complexity']:.2f})")
        
        for item in suitable_content:
            print(item)
    
    print(f"\n✅ Content complexity analysis complete!")
    print(f"🚀 Ready to build cognitive-aware recommendation engine!")

if __name__ == "__main__":
    demo_content_complexity()
