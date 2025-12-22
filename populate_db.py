"""
Script to populate the Qdrant database with textbook content.
This addresses the issue where the 'textbook_content' collection doesn't exist.
"""
import os
import requests
import xml.etree.ElementTree as ET
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configuration
SITEMAP_URL = "https://physical-ai-humanoid-textbook-mu.vercel.app/sitemap.xml"
COLLECTION_NAME = "textbook_content"  # This matches the config setting

# Initialize Cohere client
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY", "0DRQcyTI98p3HRpjuQv8tvg4IcQtRVBeublwiHoe"))

# Connect to Qdrant
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_document",  # Use search_document for indexing
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


# -------------------------------------
# Step 1 — Extract URLs from sitemap
# -------------------------------------
def get_all_urls(sitemap_url):
    xml = requests.get(sitemap_url).text
    root = ET.fromstring(xml)

    urls = []
    for child in root:
        loc_tag = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc_tag is not None:
            urls.append(loc_tag.text)

    print("\nFOUND URLS:")
    for u in urls:
        print(" -", u)

    return urls


# -------------------------------------
# Step 2 — Download page + extract text
# -------------------------------------
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        html = response.text
        text = trafilatura.extract(html)

        if not text:
            print("[WARNING] No text extracted from:", url)

        return text
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


# -------------------------------------
# Step 3 — Chunk the text
# -------------------------------------
def chunk_text(text, max_chars=1200):
    chunks = []
    while len(text) > max_chars:
        # Find a good split point (sentence boundary or paragraph)
        split_pos = text[:max_chars].rfind(". ")
        if split_pos == -1:
            split_pos = text[:max_chars].rfind("\n")
        if split_pos == -1:
            split_pos = text[:max_chars].rfind(" ")
        if split_pos == -1:
            split_pos = max_chars  # Force split at max_chars if no good boundaries found

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()  # Remove leading whitespace from remaining text
    if text:  # Add the remaining text if there's any left
        chunks.append(text)
    return chunks


# -------------------------------------
# Step 4 — Create embedding
# -------------------------------------
def embed(text):
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_document",  # Use search_document for indexing
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


# -------------------------------------
# Step 5 — Store in Qdrant
# -------------------------------------
def create_collection():
    print("\nCreating Qdrant collection...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,        # Cohere embed-english-v3.0 dimension
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully!")


def save_chunk_to_qdrant(chunk, chunk_id, url):
    vector = embed(chunk)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "url": url,
                    "content": chunk,
                    "chunk_id": chunk_id
                }
            )
        ]
    )


# -------------------------------------
# MAIN INGESTION PIPELINE
# -------------------------------------
def ingest_book():
    print("Starting ingestion process...")
    urls = get_all_urls(SITEMAP_URL)

    # Filter URLs to only include educational content (modules)
    filtered_urls = [url for url in urls if '/docs/module-' in url]
    print(f"\nFiltered to {len(filtered_urls)} educational URLs (modules only)")

    create_collection()

    global_id = 1
    total_urls = len(filtered_urls)

    for idx, url in enumerate(filtered_urls, 1):
        print(f"\n[{idx}/{total_urls}] Processing: {url}")

        try:
            text = extract_text_from_url(url)

            if not text:
                print(f"  [SKIP] No text extracted")
                continue

            chunks = chunk_text(text)
            print(f"  Created {len(chunks)} chunks")

            for ch in chunks:
                save_chunk_to_qdrant(ch, global_id, url)
                print(f"  ✓ Saved chunk {global_id}")
                global_id += 1
                time.sleep(0.1)  # Small delay to avoid rate limits

        except Exception as e:
            print(f"  [ERROR] Failed to process {url}: {e}")
            continue

        time.sleep(0.5)  # Delay between URLs

    print("\n✔️ Ingestion completed!")
    print(f"Total chunks stored: {global_id - 1}")


if __name__ == "__main__":
    print("Populating Qdrant database with textbook content...")
    ingest_book()
    print("Database population complete!")