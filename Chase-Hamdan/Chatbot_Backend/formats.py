from pydantic import BaseModel
from typing import List, Optional

class Response_Step(BaseModel):
    """Response step reasoning structured output"""
    function_call: Optional[str]
    function_output: Optional[str]
    brief_explanation: str

class Response_Reasoning(BaseModel):
    """Internal reasoning structure for response output"""
    question: str
    step_list: List[Response_Step]

class Response_Output(BaseModel):
    """Structured response format for a basic CoT step approach"""
    internal_reasoning: Response_Reasoning
    comprehensive_output_answer: str
