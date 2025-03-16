from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional

class PersonalityResponse(BaseModel):
    """Response to a question."""
    score: int = Field(
        description="Rating from 1-5 where 1=strongly disagree and 5=strongly agree",
        ge=1,  # greater than or equal to 1
        le=5   # less than or equal to 5
    )

    model_config = ConfigDict(
        json_schema_extra={
            "title": "PersonalityResponse",
            "description": "A response to a personality assessment question with a score from 1-5"
        }
    )

class ErrorResponse(BaseModel):
    """Response for cases where model evaluation failed."""
    error: str = Field(description="Error message explaining what went wrong")
    default_score: Optional[int] = Field(description="Default score used (if any)", default=None)

class ModelEvaluation(BaseModel):
    """Results from evaluating a single model."""
    model_name: str = Field(description="Name of the LLM model")
    model_version: str = Field(description="Version/API info of the model")
    responses: List[PersonalityResponse] = Field(description="List of responses to personality questions")
    errors: List[ErrorResponse] = Field(description="List of errors encountered", default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now, description="When the evaluation was conducted")

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        # Convert datetime to ISO format string
        data['timestamp'] = data['timestamp'].isoformat()
        return data

class EvaluationResults(BaseModel):
    """Complete evaluation results across all models."""
    questions: List[str] = Field(description="The personality assessment questions used")
    model_evaluations: List[ModelEvaluation] = Field(description="Results for each evaluated model") 