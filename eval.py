"""
Big Five LLM Evaluation
Evaluate LLM models' personality traits.
"""

import logging
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

# Import utils
from utils.logging_config import setup_logger
from utils.models import initialize_models
from utils.schemas import PersonalityResponse, ModelEvaluation, EvaluationResults
from utils.results_handler import save_results

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
    
    # Initialize models
    models, model_versions = initialize_models()
    
    # Configure structured output for each model
    for model_name, model in models.items():
        # Use the appropriate method for structured output
        if model_name == "GPT-4o-mini":
            models[model_name] = model.with_structured_output(PersonalityResponse, method="function_calling")
        else:
            models[model_name] = model.with_structured_output(PersonalityResponse)
    
    # Update prompt template to be more explicit about structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
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
    
    # Save results using the handler
    save_results(results.model_dump(mode='json'), questions, model_versions, logger)
    
    logger.info("Evaluation completed successfully")
    return results

if __name__ == "__main__":
    run_evaluation()
