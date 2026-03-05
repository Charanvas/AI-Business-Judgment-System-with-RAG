"""
Build RAG Index from corpus
Run on: MacBook M1 Pro
"""

import json
import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_rag_index():
    print("=" * 70)
    print("🔍 BUILDING RAG INDEX")
    print("=" * 70)
    
    # Paths
    corpus_path = Path("/Users/apple/Desktop/final/qwen-business-judgment/data/processed/rag_corpus.jsonl")
    index_dir = Path("/Users/apple/Desktop/final/qwen-business-judgment/rag_index")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Load corpus
    print(f"\n📂 Loading corpus from {corpus_path}")
    documents = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Initialize embedding model
    print("\n🤖 Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Model loaded")
    
    # Generate embeddings
    print("\n🧮 Generating embeddings...")
    texts = [doc['text'] for doc in documents]
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedding_model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings).astype('float32')
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Build FAISS index
    print("\n🔨 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"✅ Index built with {index.ntotal} vectors")
    
    # Save everything
    print("\n💾 Saving index...")
    faiss.write_index(index, str(index_dir / "faiss_index.bin"))
    
    with open(index_dir / "documents.pkl", 'wb') as f:
        pickle.dump(documents, f)
    
    np.save(index_dir / "embeddings.npy", embeddings)
    
    print(f"✅ Saved to {index_dir}")
    
    # Test retrieval
    print("\n🧪 Testing retrieval...")
    test_query = "A startup that failed due to competition and poor market fit"
    query_embedding = embedding_model.encode([test_query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, 3)
    
    print(f"\nQuery: {test_query}")
    print("\nTop 3 results:")
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        print(f"\n{i+1}. Score: {score:.4f}")
        print(f"   {documents[idx]['text'][:200]}...")
    
    print("\n" + "=" * 70)
    print("✅ RAG INDEX READY")
    print("=" * 70)

if __name__ == "__main__":
    build_rag_index()