from typing import Dict, Any, List
from src.services.base_service import BaseService


class ResponseFormatterService(BaseService):
    """
    Service for formatting agent responses according to the specified data model
    """

    def format_agent_response(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        usage_stats: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Format the agent's response according to the AgentResponse data model
        """
        try:
            formatted_response = {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
            # Add usage stats if provided
            if usage_stats:
                formatted_response["usage_stats"] = usage_stats
                
            return formatted_response
            
        except Exception as e:
            self.logger.error(f"Error formatting agent response: {str(e)}")
            raise e

    def validate_agent_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate that the agent response conforms to the data model
        """
        try:
            # Check required fields
            required_fields = ["answer", "sources", "confidence"]
            for field in required_fields:
                if field not in response:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate answer is not empty
            if not response["answer"] or not isinstance(response["answer"], str):
                self.logger.error("Answer must be a non-empty string")
                return False
            
            # Validate answer length
            if len(response["answer"]) == 0:
                self.logger.error("Answer cannot be empty")
                return False
            
            # Validate sources is a list
            if not isinstance(response["sources"], list):
                self.logger.error("Sources must be a list")
                return False
            
            # Validate confidence is between 0 and 1
            if not isinstance(response["confidence"], float) and not isinstance(response["confidence"], int):
                self.logger.error("Confidence must be a number")
                return False
            
            if response["confidence"] < 0 or response["confidence"] > 1:
                self.logger.error("Confidence must be between 0 and 1")
                return False
            
            # If usage stats exist, validate its structure
            if "usage_stats" in response:
                usage_stats = response["usage_stats"]
                if not isinstance(usage_stats, dict):
                    self.logger.error("Usage stats must be a dictionary")
                    return False
                
                required_usage_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
                for field in required_usage_fields:
                    if field not in usage_stats or not isinstance(usage_stats[field], int):
                        self.logger.error(f"Usage stats must contain integer field '{field}'")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating agent response: {str(e)}")
            return False