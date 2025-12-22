"""
Quick populate script - just populate URDF chapter for testing
"""
import os
import requests
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "textbook_content"

def chunk_text(text, max_chars=1200):
    chunks = []
    while len(text) > max_chars:
        split_pos = text[:max_chars].rfind(". ")
        if split_pos == -1:
            split_pos = text[:max_chars].rfind("\n")
        if split_pos == -1:
            split_pos = text[:max_chars].rfind(" ")
        if split_pos == -1:
            split_pos = max_chars

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    if text:
        chunks.append(text)
    return chunks

print("1. Creating collection...")
try:
    qdrant.delete_collection(COLLECTION_NAME)
except:
    pass

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)
print("[OK] Collection created")

# Test with just the URDF chapter
test_url = "https://physical-ai-humanoid-textbook-mu.vercel.app/docs/module-1/chapter-3-urdf-fundamentals"

print(f"\n2. Fetching {test_url}...")
response = requests.get(test_url, timeout=10)
text = trafilatura.extract(response.text)
print(f"[OK] Extracted {len(text)} characters")

print("\n3. Chunking text...")
chunks = chunk_text(text)
print(f"[OK] Created {len(chunks)} chunks")

print("\n4. Creating embeddings and storing...")
all_chunks_text = []
for i, chunk in enumerate(chunks, 1):
    all_chunks_text.append(chunk)
    print(f"  Chunk {i}/{len(chunks)}")

# Batch embed
print("  Creating embeddings...")
response = cohere_client.embed(
    model="embed-english-v3.0",
    input_type="search_document",
    texts=all_chunks_text,
)
embeddings = response.embeddings

# Store all at once
print("  Storing in Qdrant...")
points = []
for i, (chunk, embedding) in enumerate(zip(all_chunks_text, embeddings), 1):
    points.append(
        PointStruct(
            id=i,
            vector=embedding,
            payload={"url": test_url, "content": chunk, "chunk_id": i}
        )
    )

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"[OK] Stored {len(points)} chunks")

print("\n[SUCCESS] Quick population complete!")
print(f"You can now test the agent with questions about URDF Fundamentals")
