from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ============================================================
# STEP 1: Knowledge Base (our "documents")
# ============================================================
documents = [
    "Customers can return products within 7 days with receipt.",
    "Orders are shipped within 3 business days.",
    "Users must verify email before login.",
    "Orders can be cancelled before shipping.",
    "Premium members get free shipping on all orders.",
    "Refund is processed within 5 business days after return.",
    "Customer support is available 24/7 via chat.",
    "Password must be at least 8 characters long.",
]

print("="*60)
print("  STEP 1: Knowledge Base")
print("="*60)
print(f"  Loaded {len(documents)} document chunks\n")

# ============================================================
# STEP 2: Generate Embeddings & Build FAISS Index
# ============================================================
print("="*60)
print("  STEP 2: Building Vector Index")
print("="*60)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

print(f"  Embedding model: all-MiniLM-L6-v2")
print(f"  Vectors stored: {index.ntotal}")
print(f"  Dimensions: {dimension}\n")

# ============================================================
# STEP 3: Load LLM (HuggingFace flan-t5-base)
# ============================================================
print("="*60)
print("  STEP 3: Loading LLM (flan-t5-base)")
print("="*60)
print("  Downloading model from HuggingFace (first time only)...\n")

from transformers import T5ForConditionalGeneration, T5Tokenizer

t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

def llm(prompt_text):
    inputs = t5_tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(**inputs, max_new_tokens=150)
    return [{"generated_text": t5_tokenizer.decode(outputs[0], skip_special_tokens=True)}]

print("  LLM loaded successfully!\n")


# ============================================================
# RAG Pipeline Function
# ============================================================
def rag_query(query, top_k=2, use_grounding=True, force_chunks=None):
    """Complete RAG pipeline: query → retrieve → generate"""

    # Step A: Convert query to embedding
    query_embedding = embedding_model.encode([query])

    # Step B: Search FAISS for top-k relevant chunks
    distances, indices_result = index.search(
        np.array(query_embedding, dtype=np.float32), k=top_k
    )

    # Step C: Get retrieved chunks (or use forced chunks for experiments)
    if force_chunks:
        retrieved = force_chunks
    else:
        retrieved = [documents[i] for i in indices_result[0]]

    context = "\n".join(retrieved)

    # Step D: Build prompt
    if use_grounding:
        prompt = f"Answer the question only using the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    else:
        prompt = f"Question: {query}\n\nAnswer:"

    # Step E: Generate answer with LLM
    response = llm(prompt)[0]["generated_text"]

    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "distances": distances[0].tolist() if force_chunks is None else [],
        "prompt": prompt,
        "answer": response,
    }


def print_result(result, label=""):
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    print(f"\n  Query: \"{result['query']}\"")

    print(f"\n  Retrieved Chunks:")
    for i, chunk in enumerate(result['retrieved_chunks'], 1):
        dist = f" (distance: {result['distances'][i-1]:.4f})" if result['distances'] else ""
        print(f"    {i}. \"{chunk}\"{dist}")

    print(f"\n  LLM Answer: {result['answer']}")


# ============================================================
# MAIN EXPERIMENT: Normal RAG Pipeline
# ============================================================
print("\n" + "="*60)
print("  EXPERIMENT 1: Normal RAG Pipeline")
print("="*60)

queries = [
    "Can I get a refund?",
    "How long does shipping take?",
    "How do I create an account?",
    "Can I cancel my order?",
]

for q in queries:
    result = rag_query(q, top_k=2)
    print_result(result)

# ============================================================
# FAILURE EXPERIMENT 1: Wrong Retrieval (forced irrelevant chunks)
# ============================================================
print("\n\n" + "="*60)
print("  FAILURE EXPERIMENT 1: Wrong Retrieval")
print("="*60)
print("  Forcing irrelevant chunks into the LLM context...")

result = rag_query(
    "Can I get a refund?",
    force_chunks=["Users must verify email before login.", "Password must be at least 8 characters long."]
)
print_result(result, "Query about REFUND → but given ACCOUNT chunks")

# ============================================================
# FAILURE EXPERIMENT 2: No Grounding Instruction
# ============================================================
print("\n\n" + "="*60)
print("  FAILURE EXPERIMENT 2: No Grounding Instruction")
print("="*60)
print("  Removing 'Answer only using provided context' instruction...")

result_grounded = rag_query("Can I get a refund?", use_grounding=True)
result_ungrounded = rag_query("Can I get a refund?", use_grounding=False)

print(f"\n  Query: \"Can I get a refund?\"")
print(f"\n  WITH grounding:    {result_grounded['answer']}")
print(f"  WITHOUT grounding: {result_ungrounded['answer']}")

# ============================================================
# FAILURE EXPERIMENT 3: Too Many Chunks (high top-k)
# ============================================================
print("\n\n" + "="*60)
print("  FAILURE EXPERIMENT 3: Increasing top-k")
print("="*60)

for k in [1, 2, 4, 8]:
    result = rag_query("Can I get a refund?", top_k=min(k, len(documents)))
    print(f"\n  top-k={k}:")
    print(f"    Chunks retrieved: {[c[:40]+'...' for c in result['retrieved_chunks']]}")
    print(f"    Answer: {result['answer']}")

# ============================================================
# BONUS: Plain LLM vs RAG comparison
# ============================================================
print("\n\n" + "="*60)
print("  COMPARISON: Plain LLM vs RAG System")
print("="*60)

plain_answer = llm("Can I get a refund?")[0]["generated_text"]
rag_answer = rag_query("Can I get a refund?", top_k=2)["answer"]

print(f"\n  Query: \"Can I get a refund?\"")
print(f"\n  Plain LLM (no context):  {plain_answer}")
print(f"  RAG System (grounded):   {rag_answer}")

print("\n\n" + "="*60)
print("  SUMMARY")
print("="*60)
print("""
  What we built:
    1. Knowledge base → 8 document chunks
    2. Embedding model → all-MiniLM-L6-v2 (converts text to vectors)
    3. Vector database → FAISS (stores & searches vectors)
    4. LLM → flan-t5-base (generates answers from retrieved context)
    5. RAG pipeline → query → retrieve → generate

  What we proved:
    • Wrong retrieval → wrong answers (garbage in, garbage out)
    • No grounding instruction → LLM may hallucinate
    • Too many chunks → noise dilutes the answer
    • RAG answers are grounded; plain LLM answers are not

  The architecture:
    User Query
       ↓
    Embedding Model (all-MiniLM-L6-v2)
       ↓
    Vector Search (FAISS)
       ↓
    Retrieved Chunks (top-k)
       ↓
    Prompt Construction (context + question + grounding)
       ↓
    LLM (flan-t5-base)
       ↓
    Grounded Answer
""")
