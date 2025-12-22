"""
Simple populate script with better error handling and progress tracking
"""
import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("SIMPLE QDRANT POPULATION SCRIPT")
print("=" * 60)

# Initialize clients
print("\n[1/5] Initializing clients...")
try:
    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not all([cohere_api_key, qdrant_url, qdrant_api_key]):
        print("ERROR: Missing environment variables!")
        sys.exit(1)

    cohere_client = cohere.ClientV2(api_key=cohere_api_key)
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=False,  # Use REST API to avoid timeout issues
        timeout=30
    )
    print("[OK] Clients initialized successfully")
except Exception as e:
    print(f"ERROR initializing clients: {e}")
    sys.exit(1)

COLLECTION_NAME = "textbook_content"

# Create collection
print("\n[2/5] Creating collection...")
try:
    # Delete if exists
    try:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"  - Deleted existing collection")
    except:
        pass

    # Create new collection
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"[OK] Collection '{COLLECTION_NAME}' created")
except Exception as e:
    print(f"ERROR creating collection: {e}")
    sys.exit(1)

# Sample textbook content for testing
print("\n[3/5] Preparing sample textbook content...")
sample_content = [
    {
        "title": "ROS2 Introduction",
        "content": """ROS 2 (Robot Operating System 2) is a set of software libraries and tools for building robot applications.
        It provides a flexible framework for writing robot software. ROS 2 uses a publish-subscribe pattern for
        communication between nodes, where nodes can publish messages to topics and subscribe to topics to receive messages.
        The main components of ROS 2 include nodes, topics, services, and actions.""",
        "source": "Module 1 - Chapter 1",
        "page": 1
    },
    {
        "title": "ROS2 Nodes",
        "content": """In ROS 2, a node is a participant in the ROS graph. Nodes are the building blocks of ROS applications.
        Each node represents a single, modular purpose in the system. Nodes communicate with each other by publishing
        and subscribing to topics, calling and providing services, and using actions. A typical robot system will
        consist of many nodes working together to accomplish complex tasks.""",
        "source": "Module 1 - Chapter 1",
        "page": 2
    },
    {
        "title": "ROS2 Topics",
        "content": """Topics are named buses over which nodes exchange messages. Topics have anonymous publish/subscribe
        semantics, which decouples the production of information from its consumption. In ROS 2, topics are strongly
        typed by the ROS message type used to publish to it. Multiple publishers and subscribers can communicate
        through the same topic. Topics are identified by their name and type.""",
        "source": "Module 1 - Chapter 1",
        "page": 3
    },
    {
        "title": "URDF Basics",
        "content": """URDF (Unified Robot Description Format) is an XML format for representing a robot model in ROS.
        A URDF file specifies the robot's physical structure including links (rigid bodies) and joints (connections
        between links). Each link can have visual, collision, and inertial properties. Joints define how links move
        relative to each other and can be revolute, prismatic, fixed, continuous, planar, or floating.""",
        "source": "Module 1 - Chapter 3",
        "page": 15
    },
    {
        "title": "Humanoid Robot Structure",
        "content": """Humanoid robots are designed to resemble the human body in shape and functionality. A typical
        humanoid robot consists of a head, torso, two arms, and two legs. The robot structure is defined using URDF,
        where each body part is represented as a link, and the connections between parts are defined as joints.
        Humanoid robots typically have many degrees of freedom to enable human-like movement and interaction.""",
        "source": "Module 1 - Chapter 3",
        "page": 18
    }
]
print(f"[OK] Prepared {len(sample_content)} sample documents")

# Create embeddings
print("\n[4/5] Creating embeddings...")
try:
    texts = [doc["content"] for doc in sample_content]
    print(f"  - Embedding {len(texts)} documents...")

    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_document",
        texts=texts,
        embedding_types=["float"]
    )

    embeddings = response.embeddings.float_
    print(f"[OK] Created {len(embeddings)} embeddings")
except Exception as e:
    print(f"ERROR creating embeddings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Store in Qdrant
print("\n[5/5] Storing in Qdrant...")
try:
    points = []
    for i, (doc, embedding) in enumerate(zip(sample_content, embeddings), 1):
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "content": doc["content"],
                    "source_document": doc["source"],
                    "page_number": doc["page"],
                    "section_title": doc["title"]
                }
            )
        )
        print(f"  - Prepared point {i}/{len(sample_content)}")

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[OK] Stored {len(points)} points in Qdrant")
except Exception as e:
    print(f"ERROR storing in Qdrant: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify
print("\n[VERIFICATION]")
try:
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"[OK] Collection '{COLLECTION_NAME}' now has {info.points_count} points")
except Exception as e:
    print(f"ERROR during verification: {e}")

print("\n" + "=" * 60)
print("POPULATION COMPLETE!")
print("=" * 60)
print("\nYou can now test the chatbot with questions like:")
print("  - What is ROS 2?")
print("  - Explain ROS 2 nodes")
print("  - What is URDF?")
print("  - Tell me about humanoid robots")
