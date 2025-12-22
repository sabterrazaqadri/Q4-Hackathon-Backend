import asyncio
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from src.config.settings import settings
from src.services.agent_tools import AgentToolsService
from src.services.response_formatter import ResponseFormatterService
from src.services.base_service import BaseService


class RAGAgent(BaseService):
    """
    RAG Agent implementation using OpenAI Assistant API
    Integrates retrieval functionality as a tool and answers questions from retrieved context
    """
    
    def __init__(self):
        super().__init__()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.tools_service = AgentToolsService()
        self.response_formatter = ResponseFormatterService()
        
        # Define the tools available to the agent
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_context",
                    "description": "Retrieve relevant context from the Physical AI & Humanoid Robotics textbook based on a query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question or query to search for in the knowledge base"
                            },
                            "selected_text": {
                                "type": "string",
                                "description": "Optional selected text to provide additional context"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    async def process_query(
        self,
        question: str,
        selected_text: str = None,
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a user query using the RAG agent
        """
        try:
            self.logger.info(f"Processing query: '{question[:50]}{'...' if len(question) > 50 else ''}'")

            # First, get context using our retrieval service (this connects the retrieval service to the agent)
            retrieval_result = await self.tools_service.retrieval_tool(
                query=question,
                selected_text=selected_text
            )

            if retrieval_result.get("error"):
                self.logger.error(f"Error during retrieval: {retrieval_result['error']}")
                return self.response_formatter.format_agent_response(
                    answer="I'm sorry, I encountered an error while retrieving information. Please try again later.",
                    sources=[],
                    confidence=0.0
                )

            # Format the retrieved context for the LLM
            context_text = "\n".join(retrieval_result["documents"])

            # Create a prompt that includes the context
            if selected_text:
                prompt = f"""
                Based strictly on the following textbook content, answer the user's question.
                Do not use any information outside of this content.

                CONTEXT FROM TEXTBOOK:
                {context_text}

                USER'S SELECTED TEXT FOR CONTEXT:
                {selected_text}

                USER'S QUESTION:
                {question}

                Please provide an answer based only on the textbook content provided above.
                If the information isn't available in the context, say so.
                """
            else:
                prompt = f"""
                Based strictly on the following textbook content, answer the user's question.
                Do not use any information outside of this content.

                CONTEXT FROM TEXTBOOK:
                {context_text}

                USER'S QUESTION:
                {question}

                Please provide an answer based only on the textbook content provided above.
                If the information isn't available in the context, say so.
                """

            # Call the OpenAI API to generate a response based on the context
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "You are an assistant that answers questions based only on the provided textbook content. Do not use any information outside of the provided context. If the answer isn't in the context, clearly state that."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent factual responses
            )

            # Extract the answer from the response
            answer = response.choices[0].message.content

            # Calculate a confidence score (simplified - in reality this would be more complex)
            # For now, we'll use a fixed confidence based on whether sources were found
            confidence = 0.9 if len(retrieval_result["sources"]) > 0 else 0.1

            # Format and return the response
            formatted_response = self.response_formatter.format_agent_response(
                answer=answer,
                sources=retrieval_result["sources"],
                confidence=confidence,
                usage_stats={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )

            # Validate the response before returning
            if not self.response_formatter.validate_agent_response(formatted_response):
                self.logger.error("Formatted response failed validation")
                return self.response_formatter.format_agent_response(
                    answer="I'm sorry, there was an issue formatting the response. Please try again later.",
                    sources=[],
                    confidence=0.0
                )

            # Validate that the response is properly grounded in the context
            is_valid = await self.validate_response_grounding(
                question=question,
                answer=answer,
                sources=retrieval_result["sources"]
            )

            if not is_valid:
                self.logger.warning("Response failed grounding validation")
                return self.response_formatter.format_agent_response(
                    answer="I'm sorry, I couldn't find sufficient relevant information to answer your question.",
                    sources=[],
                    confidence=0.3  # Lower confidence for ungrounded answer
                )

            return formatted_response

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return self.response_formatter.format_agent_response(
                answer="I'm sorry, I encountered an error while processing your question. Please try again later.",
                sources=[],
                confidence=0.0
            )

    async def validate_response_grounding(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate that the response is properly grounded in the retrieved context
        """
        try:
            # For now, implement a simple check that sources were provided with non-empty answer
            # A more sophisticated implementation would check semantic alignment
            has_sources = len(sources) > 0
            has_answer = len(answer.strip()) > 0
            
            is_valid = has_sources and has_answer
            
            if not is_valid:
                self.logger.warning("Response validation failed: missing sources or answer")
            
            return is_valid
        except Exception as e:
            self.logger.error(f"Error validating response grounding: {str(e)}")
            return False