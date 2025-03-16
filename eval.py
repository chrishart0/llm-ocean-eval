"""
Big Five LLM Evaluation
Run tests to evaluate personality traits of GPT-4o-mini, Claude 3, and Grok.
"""

import os
import logging
import colorlog
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from settings import settings
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import ChatPromptTemplate

# Update Pydantic models with V2 syntax
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

class ModelEvaluation(BaseModel):
    """Results from evaluating a single model."""
    model_name: str = Field(description="Name of the LLM model")
    model_version: str = Field(description="Version/API info of the model")
    responses: List[PersonalityResponse] = Field(description="List of responses to personality questions")
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

def setup_logger():
    """Set up and configure the logger."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logger
    log_file = f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Set up color formatter for console
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
            'AI_RESPONSE': 'green',  # Custom level for AI responses
        },
        secondary_log_colors={},
        style='%'
    )
    
    # Set up regular formatter for file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Set up logger
    logger = logging.getLogger("big_five_eval")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Add custom log level for AI responses
    logging.AI_RESPONSE = 25  # Between INFO and WARNING
    logging.addLevelName(logging.AI_RESPONSE, 'AI_RESPONSE')
    
    def ai_response(self, message, *args, **kwargs):
        self.log(logging.AI_RESPONSE, message, *args, **kwargs)
    
    logging.Logger.ai_response = ai_response
    
    logger.info("Logger initialized")
    return logger


def load_bfi_questions():
    """Load the BFI questions from the prompts file."""
    logger = logging.getLogger("big_five_eval")
    prompts_path = Path("prompts/bfi_subset.txt")
    
    logger.info(f"Loading BFI questions from {prompts_path}")
    try:
        with open(prompts_path, "r") as f:
            questions = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(questions)} questions")
        return questions
    except Exception as e:
        logger.error(f"Error loading questions: {str(e)}")
        raise


def run_evaluation():
    """Run the evaluation on different LLM models."""
    logger = setup_logger()
    logger.info("Starting evaluation")
    questions = load_bfi_questions()
    
    # Initialize models with structured output and function calling
    models = {}
    model_versions = {}
    
    logger.info("Initializing models")
    
    if settings.openai_api_key:
        logger.info("Initializing GPT-4o-mini")
        models["GPT-4o-mini"] = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.openai_api_key
        ).with_structured_output(PersonalityResponse, method="function_calling")
        model_versions["GPT-4o-mini"] = "OpenAI API, gpt-4o-mini"
    else:
        logger.warning("OpenAI API key not found, skipping GPT-4o-mini")
    
    if settings.anthropic_api_key:
        logger.info("Initializing Claude 3")
        models["Claude 3"] = ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0,
            api_key=settings.anthropic_api_key
        ).with_structured_output(PersonalityResponse)
        model_versions["Claude 3"] = "Anthropic API, claude-3-opus-20240229"
    else:
        logger.warning("Anthropic API key not found, skipping Claude 3")
    
    if settings.xai_api_key:
        logger.info("Initializing Grok")
        models["Grok"] = ChatXAI(
            model="grok-beta",
            temperature=0,
            api_key=settings.xai_api_key
        ).with_structured_output(PersonalityResponse)
        model_versions["Grok"] = "xAI API, grok-beta"
    else:
        logger.warning("xAI API key not found, skipping Grok")
    
    # Update prompt template to be more explicit about structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are evaluating personality traits. For each statement, provide a numerical rating that best describes your level of agreement.

        You must respond with a score from 1-5 where:
        1 = Strongly Disagree
        2 = Disagree
        3 = Neutral
        4 = Agree
        5 = Strongly Agree

        Provide only the numerical score that best matches your response."""),
        ("human", "{question}")
    ])
    
    # Store results in our structured format
    all_evaluations = []
    
    # Run evaluation for each model
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        responses = []
        
        for question in questions:
            question_text = question.split(" (Tests")[0] if " (Tests" in question else question
            logger.info(f"Sending question to {model_name}: '{question_text}'")
            
            try:
                chain = prompt | model
                response = chain.invoke({"question": question_text})
                logger.ai_response(f"Response from {model_name}: {response}")
                responses.append(response)
            except Exception as e:
                logger.error(f"Error querying {model_name}: {str(e)}")
                # Add a null response on error
                responses.append(PersonalityResponse(score=0))
        
        # Create structured evaluation for this model
        model_eval = ModelEvaluation(
            model_name=model_name,
            model_version=model_versions[model_name],
            responses=responses
        )
        all_evaluations.append(model_eval)
    
    # Create final structured results
    results = EvaluationResults(
        questions=questions,
        model_evaluations=all_evaluations
    )
    
    # Save results using the existing handler - use model_dump() to properly serialize
    from results_handler import save_results
    save_results(results.model_dump(mode='json'), questions, model_versions, logger)
    
    logger.info("Evaluation completed successfully")
    return results


if __name__ == "__main__":
    run_evaluation()
