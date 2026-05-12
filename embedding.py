from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Same document from chunking.py ---
document = """
Refund Policy:
Customers can return products within 7 days with receipt.

Shipping Policy:
Orders are shipped within 3 business days.

Account Setup:
Users must verify email before login.

Cancellation Policy:
Orders can be cancelled before shipping.
"""

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')
query = "Can I return a product?"
query_embedding = model.encode([query])


# --- Helper: run retrieval and print scores ---
def retrieve_and_score(chunks, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i+1}: {chunk[:80]}{'...' if len(chunk) > 80 else ''}")

    embeddings = model.encode(chunks)
    scores = cosine_similarity(query_embedding, embeddings)[0]

    print(f"\n  Query: \"{query}\"")
    print(f"  {'─'*40}")
    for i, score in enumerate(scores):
        marker = " ← BEST" if score == max(scores) else ""
        print(f"  Chunk {i+1}: {score:.4f}{marker}")

    best_idx = scores.argmax()
    print(f"\n  Retrieved: Chunk {best_idx+1} → {chunks[best_idx][:60]}...")
    return scores


# ============================================================
# EXPERIMENT 1: Semantic Chunking (paragraph-based)
# ============================================================
semantic_chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]
retrieve_and_score(semantic_chunks, "EXPERIMENT 1: Semantic Chunking (by paragraph)")


# ============================================================
# EXPERIMENT A: Very Large Chunks (noisy retrieval)
# ============================================================
large_chunks = [
    # Combine Refund + Shipping + Account into one giant chunk
    "Refund Policy:\nCustomers can return products within 7 days with receipt.\n\n"
    "Shipping Policy:\nOrders are shipped within 3 business days.\n\n"
    "Account Setup:\nUsers must verify email before login.",
    # Cancellation alone
    "Cancellation Policy:\nOrders can be cancelled before shipping.",
]
retrieve_and_score(large_chunks, "EXPERIMENT A: Very Large Chunks (3 topics merged)")


# ============================================================
# EXPERIMENT B: Tiny Chunks (sentence-by-sentence)
# ============================================================
tiny_chunks = [
    line.strip()
    for line in document.split("\n")
    if line.strip()
]
retrieve_and_score(tiny_chunks, "EXPERIMENT B: Tiny Chunks (sentence-by-sentence)")


# ============================================================
# EXPERIMENT C: Overlap Chunks
# ============================================================
def overlap_chunk(text, chunk_size=100, overlap=30):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap  # step back by overlap amount
    return [c for c in chunks if c]

overlap_chunks = overlap_chunk(document)
retrieve_and_score(overlap_chunks, "EXPERIMENT C: Overlap Chunks (size=100, overlap=30)")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print("""
  Semantic Chunking  → Best precision. Each chunk = 1 topic.
                        Retrieval finds the exact policy.

  Large Chunks       → Noisy. The refund answer is buried with
                        shipping + account info. Score is diluted.

  Tiny Chunks        → Fragmented. "Refund Policy:" and its answer
                        are separate chunks. Meaning is split.

  Overlap Chunks     → Better continuity than fixed chunks.
                        Context is preserved across boundaries.

  KEY INSIGHT: Chunking directly controls retrieval precision
               and context preservation.
""")