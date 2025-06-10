"""
Basic test of LLM functionality
"""

# Test 1: Basic imports
print("Testing imports...")
try:
    import pandas as pd
    import numpy as np
    print("✅ Pandas/NumPy working")
except Exception as e:
    print(f"❌ Pandas/NumPy error: {e}")

# Test 2: Ollama connection
print("\nTesting Ollama...")
try:
    import ollama
    response = ollama.chat(
        model='llama2',
        messages=[{'role': 'user', 'content': 'Say "working" if you can respond'}]
    )
    print(f"✅ Ollama working: {response['message']['content']}")
except Exception as e:
    print(f"❌ Ollama error: {e}")
    print("💡 Try: ollama pull llama2")

# Test 3: Sentence transformers
print("\nTesting sentence transformers...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode("test sentence")
    print(f"✅ Embeddings working: {len(embedding)} dimensions")
except Exception as e:
    print(f"❌ Embeddings error: {e}")

# Test 4: ChromaDB
print("\nTesting ChromaDB...")
try:
    import chromadb
    client = chromadb.Client()
    print("✅ ChromaDB working")
except Exception as e:
    print(f"❌ ChromaDB error: {e}")

print("\n🎉 Basic test complete!")
