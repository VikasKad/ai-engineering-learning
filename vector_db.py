from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Step 1: Our document chunks ---
chunks = [
    "Customers can return products within 7 days.",
    "Orders are shipped within 3 business days.",
    "Users must verify email before login.",
    "Orders can be cancelled before shipping.",
]

# --- Step 2: Generate embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

print(f"Embeddings shape: {embeddings.shape}")
print(f"  → {len(chunks)} chunks, each represented as {embeddings.shape[1]} dimensions\n")

# --- Step 3: Create FAISS index and store embeddings ---
dimension = embeddings.shape[1]  # 384 for MiniLM-L6

# IndexFlatL2 = exact search using L2 (Euclidean) distance
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

print(f"FAISS index created!")
print(f"  → Total vectors stored: {index.ntotal}")
print(f"  → Dimension: {dimension}")

# --- Step 4: Query the vector database ---
print("\n" + "="*60)
print("  QUERY TEST")
print("="*60)

query = "Can I return an item?"
query_embedding = model.encode([query])

# Search for top-k nearest vectors
k = 2  # retrieve top 2 most similar chunks
distances, indices = index.search(
    np.array(query_embedding, dtype=np.float32),
    k=k
)

print(f"\n  Query: \"{query}\"")
print(f"  Top {k} results:\n")

for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    print(f"  #{rank}: \"{chunks[idx]}\"")
    print(f"      Distance: {dist:.4f} (lower = more similar)\n")

# --- Step 5: Try multiple queries ---
print("="*60)
print("  MULTIPLE QUERIES")
print("="*60)

queries = [
    "How do I get my money back?",
    "When will my order arrive?",
    "How to create an account?",
    "I want to cancel my purchase",
]

for q in queries:
    q_emb = model.encode([q])
    distances, indices = index.search(
        np.array(q_emb, dtype=np.float32), k=1
    )
    best_idx = indices[0][0]
    best_dist = distances[0][0]
    print(f"\n  Query: \"{q}\"")
    print(f"  → Retrieved: \"{chunks[best_idx]}\" (distance: {best_dist:.4f})")

# --- Step 6: Scale comparison ---
print("\n\n" + "="*60)
print("  WHY VECTOR DB MATTERS: SCALE COMPARISON")
print("="*60)

import time

# Simulate larger dataset by duplicating chunks
large_chunks = chunks * 2500  # 10,000 chunks
large_embeddings = np.tile(embeddings, (2500, 1)).astype(np.float32)

# Method 1: Brute force cosine similarity (what we did before)
from sklearn.metrics.pairwise import cosine_similarity

start = time.time()
scores = cosine_similarity(query_embedding, large_embeddings)
brute_time = time.time() - start

# Method 2: FAISS index search
large_index = faiss.IndexFlatL2(dimension)
large_index.add(large_embeddings)

start = time.time()
distances, indices = large_index.search(
    np.array(query_embedding, dtype=np.float32), k=2
)
faiss_time = time.time() - start

print(f"\n  Dataset: {len(large_chunks)} chunks")
print(f"  Brute force (sklearn):  {brute_time*1000:.2f} ms")
print(f"  FAISS index search:     {faiss_time*1000:.2f} ms")
print(f"\n  Note: Both are exact search here. The real speedup comes")
print(f"  with ANN (approximate) indexes on millions of vectors.")

print("\n\n" + "="*60)
print("  SUMMARY")
print("="*60)
print("""
  What we did:
    1. Encoded chunks into embeddings (384-dim vectors)
    2. Stored them in a FAISS index
    3. Queried with natural language → got semantically similar chunks
    4. Compared scale performance

  Key concepts:
    • Vector DB stores VECTORS (numbers), not text
    • Search finds nearest vectors by distance
    • Lower distance = more similar meaning
    • IndexFlatL2 = exact search (good for learning)
    • For millions of vectors → use ANN indexes (faster, slight accuracy tradeoff)
""")
