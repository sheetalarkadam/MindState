"""
RAG-Based Content Analysis System
Uses vector embeddings and retrieval for deep content understanding
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import json
import asyncio
from datetime import datetime
import sys
import os

# LLM imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class RAGContentAnalyzer:
    """Advanced content analysis using Retrieval-Augmented Generation"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize vector database
        self.chroma_client = chromadb.Client()
        
        # Create collections for different content types
        try:
            self.content_collection = self.chroma_client.create_collection(
                name="content_embeddings",
                metadata={"description": "Content embeddings for semantic analysis"}
            )
            self.cognitive_patterns_collection = self.chroma_client.create_collection(
                name="cognitive_patterns",
                metadata={"description": "Cognitive load patterns and examples"}
            )
        except:
            # Collections might already exist
            self.content_collection = self.chroma_client.get_collection("content_embeddings")
            self.cognitive_patterns_collection = self.chroma_client.get_collection("cognitive_patterns")
        
        # Initialize cognitive load knowledge base
        self.init_cognitive_knowledge_base()
    
    def init_cognitive_knowledge_base(self):
        """Initialize the knowledge base with cognitive load patterns"""
        
        cognitive_examples = [
            {
                "content_type": "low_complexity",
                "description": "Simple comedies with familiar characters and predictable plots",
                "examples": "Friends, The Office, sitcoms with laugh tracks",
                "cognitive_load": 0.2,
                "optimal_states": ["stress_state", "recovery_mode"],
                "attention_required": "minimal sustained attention",
                "processing_demands": "pattern recognition, emotional comfort"
            },
            {
                "content_type": "medium_complexity", 
                "description": "Moderate complexity with clear narratives and some learning elements",
                "examples": "Nature documentaries, light dramas, adventure films",
                "cognitive_load": 0.5,
                "optimal_states": ["discovery_mode"],
                "attention_required": "moderate sustained attention",
                "processing_demands": "information integration, visual processing"
            },
            {
                "content_type": "high_complexity",
                "description": "Complex narratives requiring active thinking and analysis",
                "examples": "Inception, Westworld, philosophical content",
                "cognitive_load": 0.8,
                "optimal_states": ["flow_state", "learning_mode"],
                "attention_required": "high sustained attention",
                "processing_demands": "working memory, analytical thinking, pattern synthesis"
            },
            {
                "content_type": "educational",
                "description": "Learning-focused content with structured information",
                "examples": "Tutorials, documentaries, educational series",
                "cognitive_load": 0.7,
                "optimal_states": ["learning_mode"],
                "attention_required": "focused attention with note-taking",
                "processing_demands": "information encoding, concept mapping"
            },
            {
                "content_type": "interactive_high",
                "description": "Interactive content requiring decision-making and multitasking",
                "examples": "Strategy games, complex simulations",
                "cognitive_load": 0.9,
                "optimal_states": ["flow_state"],
                "attention_required": "divided attention, rapid decision making",
                "processing_demands": "executive function, working memory, motor control"
            }
        ]
        
        # Add to vector database
        for i, example in enumerate(cognitive_examples):
            # Create embedding from description and characteristics
            text_for_embedding = f"{example['description']} {example['examples']} {example['attention_required']} {example['processing_demands']}"
            
            try:
                self.cognitive_patterns_collection.add(
                    embeddings=[self.embedding_model.encode(text_for_embedding).tolist()],
                    documents=[json.dumps(example)],
                    metadatas=[{
                        "content_type": example["content_type"],
                        "cognitive_load": example["cognitive_load"]
                    }],
                    ids=[f"pattern_{i}"]
                )
            except:
                pass  # Might already exist
    
    def embed_content(self, content: Dict) -> List[float]:
        """Create semantic embedding for content"""
        
        # Combine all textual elements
        content_text = f"""
        Title: {content.get('title', '')}
        Genre: {content.get('genre', '')}
        Description: {content.get('description', '')}
        Plot: {content.get('plot', '')}
        Type: {content.get('type', '')}
        Duration: {content.get('duration', 0)} minutes
        """
        
        return self.embedding_model.encode(content_text.strip()).tolist()
    
    def retrieve_similar_patterns(self, content_embedding: List[float], n_results: int = 3) -> List[Dict]:
        """Retrieve similar cognitive load patterns from knowledge base"""
        
        results = self.cognitive_patterns_collection.query(
            query_embeddings=[content_embedding],
            n_results=n_results
        )
        
        patterns = []
        for i, doc in enumerate(results['documents'][0]):
            pattern = json.loads(doc)
            pattern['similarity_score'] = 1 - results['distances'][0][i]  # Convert distance to similarity
            patterns.append(pattern)
        
        return patterns
    
    async def analyze_with_rag_llm(self, content: Dict, similar_patterns: List[Dict]) -> Dict:
        """Use RAG + LLM for sophisticated content analysis"""
        
        # Create context from retrieved patterns
        context = "COGNITIVE LOAD PATTERNS FROM KNOWLEDGE BASE:\n"
        for pattern in similar_patterns:
            context += f"- {pattern['content_type']}: {pattern['description']} (cognitive load: {pattern['cognitive_load']}, similarity: {pattern['similarity_score']:.2f})\n"
        
        prompt = f"""
{context}

CONTENT TO ANALYZE:
Title: {content.get('title', 'Unknown')}
Genre: {content.get('genre', 'Unknown')}
Type: {content.get('type', 'Unknown')}
Duration: {content.get('duration', 0)} minutes
Description: {content.get('description', '')}
Plot Summary: {content.get('plot', '')}

TASK: Analyze this content's cognitive complexity using the knowledge base patterns as reference.

Consider these factors:
1. Narrative complexity (plot structure, character development)
2. Conceptual difficulty (abstract concepts, technical content)
3. Attention requirements (sustained focus needed)
4. Processing demands (working memory, analytical thinking)
5. Emotional intensity (stress, excitement levels)
6. Information density (how much information per minute)

Provide analysis in this JSON format:
{{
    "overall_complexity": 0.XX,
    "complexity_category": "low|medium|high",
    "cognitive_dimensions": {{
        "narrative_complexity": 0.XX,
        "conceptual_difficulty": 0.XX,
        "attention_demands": 0.XX,
        "processing_load": 0.XX,
        "emotional_intensity": 0.XX,
        "information_density": 0.XX
    }},
    "optimal_cognitive_states": ["state1", "state2"],
    "reasoning": "detailed explanation of complexity assessment",
    "attention_span_required": "estimated minutes of sustained attention",
    "cognitive_load_factors": [
        "specific factor that contributes to cognitive load",
        "another contributing factor"
    ],
    "recommendations": {{
        "stress_state": "recommendation for stressed users",
        "recovery_mode": "recommendation for recovering users", 
        "discovery_mode": "recommendation for exploring users",
        "learning_mode": "recommendation for learning users",
        "flow_state": "recommendation for focused users"
    }}
}}
"""
        
        if OLLAMA_AVAILABLE:
            try:
                response = ollama.chat(
                    model='llama2',
                    messages=[
                        {'role': 'system', 'content': 'You are an expert in cognitive psychology and media analysis. Respond only with valid JSON.'},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                
                result_text = response['message']['content']
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                json_str = result_text[start_idx:end_idx]
                
                return json.loads(json_str)
                
            except Exception as e:
                print(f"LLM analysis failed: {e}")
                return self._fallback_rag_analysis(content, similar_patterns)
        else:
            return self._fallback_rag_analysis(content, similar_patterns)
    
    def _fallback_rag_analysis(self, content: Dict, similar_patterns: List[Dict]) -> Dict:
        """Fallback analysis using retrieved patterns"""
        
        # Use similarity to estimate complexity
        if similar_patterns:
            avg_complexity = np.mean([p['cognitive_load'] for p in similar_patterns])
            closest_pattern = similar_patterns[0]
        else:
            avg_complexity = 0.5
            closest_pattern = {"optimal_states": ["discovery_mode"]}
        
        return {
            "overall_complexity": avg_complexity,
            "complexity_category": "low" if avg_complexity < 0.4 else "medium" if avg_complexity < 0.7 else "high",
            "cognitive_dimensions": {
                "narrative_complexity": avg_complexity,
                "conceptual_difficulty": avg_complexity * 0.9,
                "attention_demands": avg_complexity * 1.1,
                "processing_load": avg_complexity,
                "emotional_intensity": avg_complexity * 0.8,
                "information_density": avg_complexity * 0.7
            },
            "optimal_cognitive_states": closest_pattern.get("optimal_states", ["discovery_mode"]),
            "reasoning": f"Analysis based on similarity to {len(similar_patterns)} patterns from knowledge base",
            "attention_span_required": f"{int(avg_complexity * 60)} minutes",
            "cognitive_load_factors": ["pattern-based analysis", "semantic similarity"],
            "recommendations": {
                "stress_state": "Consider only if complexity < 0.3",
                "recovery_mode": "Suitable if complexity < 0.4",
                "discovery_mode": "Good fit for most complexity levels",
                "learning_mode": "Ideal if complexity > 0.6",
                "flow_state": "Perfect if complexity > 0.7"
            }
        }
    
    async def analyze_content_with_rag(self, content: Dict) -> Dict:
        """Main method: Analyze content using RAG approach"""
        
        # 1. Create content embedding
        content_embedding = self.embed_content(content)
        
        # 2. Retrieve similar patterns
        similar_patterns = self.retrieve_similar_patterns(content_embedding)
        
        # 3. Analyze with LLM + RAG
        rag_analysis = await self.analyze_with_rag_llm(content, similar_patterns)
        
        # 4. Store content embedding for future retrieval
        try:
            self.content_collection.add(
                embeddings=[content_embedding],
                documents=[json.dumps(content)],
                metadatas=[{
                    "title": content.get("title", ""),
                    "genre": content.get("genre", ""),
                    "complexity": rag_analysis.get("overall_complexity", 0.5)
                }],
                ids=[f"content_{hash(content.get('title', '') + str(datetime.now()))}"]
            )
        except:
            pass  # Handle duplicate IDs
        
        # 5. Enhance with semantic search capabilities
        rag_analysis['semantic_analysis'] = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'similar_patterns_found': len(similar_patterns),
            'knowledge_base_enhanced': True,
            'retrieval_confidence': np.mean([p['similarity_score'] for p in similar_patterns]) if similar_patterns else 0.5
        }
        
        return rag_analysis
    
    def find_similar_content(self, target_content: Dict, n_results: int = 5) -> List[Dict]:
        """Find content similar to target using semantic search"""
        
        target_embedding = self.embed_content(target_content)
        
        try:
            results = self.content_collection.query(
                query_embeddings=[target_embedding],
                n_results=n_results
            )
            
            similar_content = []
            for i, doc in enumerate(results['documents'][0]):
                content = json.loads(doc)
                content['similarity_score'] = 1 - results['distances'][0][i]
                similar_content.append(content)
            
            return similar_content
        except:
            return []

def demo_rag_content_analysis():
    """Demonstrate RAG-based content analysis"""
    
    print("🔍 RAG-Based Content Analysis Demo")
    print("=" * 50)
    
    # Initialize RAG analyzer
    analyzer = RAGContentAnalyzer()
    
    # Sample content for analysis
    test_content = [
        {
            'title': 'Inception',
            'genre': 'Sci-Fi/Thriller',
            'type': 'movie',
            'duration': 148,
            'description': 'A thief who steals corporate secrets through dream-sharing technology.',
            'plot': 'Complex multi-layered narrative with dreams within dreams, time dilation, and philosophical themes about reality and consciousness.'
        },
        {
            'title': 'Friends - The One with the Embryos',
            'genre': 'Comedy/Sitcom',
            'type': 'tv_series', 
            'duration': 22,
            'description': 'Classic sitcom episode with beloved characters.',
            'plot': 'Light-hearted trivia game between friends with predictable humor and familiar character dynamics.'
        },
        {
            'title': 'Cosmos: A Personal Voyage',
            'genre': 'Documentary/Science',
            'type': 'documentary',
            'duration': 60,
            'description': 'Carl Sagan explores the universe and scientific discovery.',
            'plot': 'Educational journey through space and time, covering complex astrophysics concepts with philosophical implications.'
        }
    ]
    
    print("🧠 Analyzing content with RAG + LLM...")
    
    for content in test_content:
        print(f"\n📺 Analyzing: {content['title']}")
        print("-" * 40)
        
        # Analyze with RAG
        import asyncio
        analysis = asyncio.run(analyzer.analyze_content_with_rag(content))
        
        print(f"🎯 Overall Complexity: {analysis['overall_complexity']:.2f} ({analysis['complexity_category']})")
        print(f"🧠 Optimal States: {', '.join(analysis['optimal_cognitive_states'])}")
        print(f"⏱️ Attention Required: {analysis['attention_span_required']}")
        print(f"📊 RAG Confidence: {analysis['semantic_analysis']['retrieval_confidence']:.2f}")
        print(f"💭 Reasoning: {analysis['reasoning'][:100]}...")
        
        # Show cognitive dimensions
        dimensions = analysis['cognitive_dimensions']
        print(f"\n📈 Cognitive Dimensions:")
        for dim, score in dimensions.items():
            print(f"   {dim}: {score:.2f}")
    
    print(f"\n🎉 RAG Content Analysis Demo Complete!")
    print(f"🚀 Advantages over traditional analysis:")
    print(f"   • Semantic understanding through embeddings")
    print(f"   • Knowledge base of cognitive patterns")
    print(f"   • LLM reasoning with retrieved context")
    print(f"   • Similarity search for content discovery")

if __name__ == "__main__":
    demo_rag_content_analysis()
