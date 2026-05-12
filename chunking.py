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
# print(document)

def fixed_chunk(text, chunk_size=80):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

# chunks = fixed_chunk(document)

# for i, chunk in enumerate(chunks):
#     print(f"\nChunk {i+1}:")
#     print(chunk)

semantic_chunks = [
    chunk.strip()
    for chunk in document.split("\n\n")
    if chunk.strip()
]

for i, chunk in enumerate(semantic_chunks):
    print(f"\nSemantic Chunk {i+1}:")
    print(chunk)