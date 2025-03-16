"""
Big Five LLM Evaluation
Evaluate LLM models' personality traits.
"""

import logging
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
import argparse
import yaml

# Import utils
from utils.logging_config import setup_logger
from utils.models import initialize_models, ModelRegistry
from utils.schemas import PersonalityResponse, ModelEvaluation, EvaluationResults, ErrorResponse
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

def run_evaluation(selected_model=None, batch_file=None):
    """Run the evaluation on different LLM models."""
    logger = setup_logger()
    logger.info("Starting evaluation")
    questions = load_bfi_questions()
    
    # Initialize models (with optional selection)
    models, model_versions = initialize_models(selected_model, batch_file)
    
    if not models:
        logger.error("No models available for evaluation. Exiting.")
        return None
    
    # Load batch parameters if available
    batch_params = {}
    if batch_file:
        try:
            with open(batch_file, "r") as f:
                batch_config = yaml.safe_load(f)
                batch_params = batch_config.get("parameters", {})
                logger.info(f"Loaded batch parameters: {batch_params}")
                
                # Apply max questions parameter if set
                max_questions = batch_params.get("max_questions_per_batch")
                if max_questions and max_questions < len(questions):
                    logger.info(f"Limiting to {max_questions} questions as specified in batch config")
                    questions = questions[:max_questions]
        except Exception as e:
            logger.error(f"Error loading batch parameters: {str(e)}")
    
    # Configure structured output for each model
    for model_name, model in models.items():
        # Get the structured output method from batch parameters or use default
        method = batch_params.get("structured_output_method", "default")
        
        # If it's an OpenAI model, always use function_calling
        if "GPT" in model_name:
            logger.info(f"Using function_calling method for {model_name}")
            models[model_name] = model.with_structured_output(PersonalityResponse, method="function_calling")
        else:
            logger.info(f"Using default structured output method for {model_name}")
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
    
    # Get default error score from batch parameters
    default_error_score = batch_params.get("default_error_score", 3)  # Use 3 (neutral) as default if not specified
    retry_failed = batch_params.get("retry_failed", False)
    
    # Run evaluation for each model
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        responses = []
        errors = []
        
        for i, question in enumerate(questions):
            question_text = question.split(" (Tests")[0] if " (Tests" in question else question
            logger.info(f"Sending question to {model_name}: '{question_text}'")
            
            try:
                chain = prompt | model
                response = chain.invoke({"question": question_text})
                logger.ai_response(f"Response from {model_name}: {response}")
                responses.append(response)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error querying {model_name}: {error_msg}")
                
                # Record the error
                errors.append(ErrorResponse(
                    error=error_msg,
                    default_score=default_error_score if default_error_score else None
                ))
                
                # Try again if retry is enabled
                if retry_failed:
                    logger.info(f"Retrying question {i+1} for {model_name}...")
                    try:
                        response = chain.invoke({"question": question_text})
                        logger.ai_response(f"Retry response from {model_name}: {response}")
                        responses.append(response)
                        # Remove the error since we succeeded on retry
                        errors.pop()
                        continue
                    except Exception as retry_e:
                        logger.error(f"Retry also failed: {str(retry_e)}")
                
                # If we get here, either we didn't retry or the retry failed
                # If default_error_score is set, use it (must be 1-5)
                if default_error_score and 1 <= default_error_score <= 5:
                    try:
                        responses.append(PersonalityResponse(score=default_error_score))
                        logger.info(f"Using default error score: {default_error_score}")
                    except Exception as score_e:
                        logger.error(f"Error creating default response: {str(score_e)}")
                        # If we can't create a valid response, ensure lists stay in sync
                        if len(responses) < i:
                            # Add None to keep the indices aligned with questions
                            responses.append(None)
                else:
                    # If no default score or invalid default, ensure lists stay in sync
                    if len(responses) < i:
                        # Add None to keep the indices aligned with questions
                        responses.append(None)
        
        # Filter out None values from responses
        valid_responses = [r for r in responses if r is not None]
        
        # Create structured evaluation for this model
        model_eval = ModelEvaluation(
            model_name=model_name,
            model_version=model_versions[model_name],
            responses=valid_responses,
            errors=errors
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
    parser = argparse.ArgumentParser(description="Run Big Five personality evaluation on LLMs")
    
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, help="Specific model to evaluate (format: 'provider:model_id')")
    model_group.add_argument("--batch", type=str, help="Path to batch configuration file")
    
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        registry = ModelRegistry()
        available_models = registry.list_available_models()
        print("Available models:")
        for model in available_models:
            print(f" - {model}")
        exit(0)
    
    run_evaluation(selected_model=args.model, batch_file=args.batch)
