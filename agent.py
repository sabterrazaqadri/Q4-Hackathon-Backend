from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents import set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv


load_dotenv()
set_tracing_disabled(disabled=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# gemini_api_key = os.getenv("GEMINI_API_KEY")
# provider = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=provider
# )

# Initialize OpenAI model
model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY)
)

import cohere
from qdrant_client import QdrantClient

# Initialize Cohere client
cohere_client = cohere.Client("0DRQcyTI98p3HRpjuQv8tvg4IcQtRVBeublwiHoe")
# Connect to Qdrant
qdrant = QdrantClient(
    url="https://6eb3cc7d-3f4e-46a5-ae7c-20d8d583c238.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4m9T9I-NlGbJF6KZG0edJ4FS2xfOoYMCSlGYVbv-Mss",
)



def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding


@function_tool
def retrieve(query: str) -> list[str]:
    """Retrieve relevant content from the textbook based on the query."""
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="textbook_content",
        query=embedding,
        limit=5
    )
    return [point.payload.get("content", point.payload.get("text", "")) for point in result.points]




agent = Agent(
    name="Assistant",
    model=model,
    instructions="""
You are an AI tutor for the Physical AI & Humanoid Robotics textbook.
To answer the user question, first call the tool `retrieve` with the user query.
Use ONLY the returned content from `retrieve` to answer.
If the answer is not in the retrieved content, say "I don't know".
""",
    tools=[retrieve]
)


result = Runner.run_sync(
    agent,
    input="what is URDF Fundamentals?",
)

print(result.final_output)