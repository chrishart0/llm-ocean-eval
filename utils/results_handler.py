"""Handle saving and loading evaluation results."""

import os
import json
from datetime import datetime
import csv
import logging


def save_results(results_data, questions, model_versions, logger):
    """Save evaluation results to JSON and CSV formats."""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full results as JSON
    json_filename = f"results/evaluation_{timestamp}.json"
    logger.info(f"Saving full results to {json_filename}")
    
    with open(json_filename, "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Save scores to CSV for easier analysis
    csv_filename = f"results/scores_{timestamp}.csv"
    logger.info(f"Saving scores to {csv_filename}")
    
    model_names = [eval_data["model_name"] for eval_data in results_data["model_evaluations"]]
    
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["Question"] + model_names
        writer.writerow(header)
        
        # Write scores for each question
        for i, question in enumerate(questions):
            # Extract question text from different possible formats
            if isinstance(question, dict) and 'question' in question:
                question_text = question['question']
            elif hasattr(question, 'question'):
                question_text = question.question
            else:
                question_text = str(question)
                
            row = [question_text]
            for eval_data in results_data["model_evaluations"]:
                try:
                    if i < len(eval_data["responses"]):
                        score = eval_data["responses"][i]["score"]
                        row.append(score)
                    else:
                        # Check if there's an error for this question
                        error_index = min(i, len(eval_data.get("errors", [])) - 1)
                        if error_index >= 0 and "default_score" in eval_data["errors"][error_index]:
                            row.append(eval_data["errors"][error_index]["default_score"])
                        else:
                            row.append("N/A")
                except (IndexError, KeyError):
                    row.append("N/A")
            writer.writerow(row)
    
    # Save errors to a separate CSV
    error_filename = f"results/errors_{timestamp}.csv"
    has_errors = any(len(eval_data.get("errors", [])) > 0 for eval_data in results_data["model_evaluations"])
    
    if has_errors:
        logger.info(f"Saving errors to {error_filename}")
        with open(error_filename, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["Model", "Question", "Error"])
            
            # Write errors
            for eval_data in results_data["model_evaluations"]:
                model_name = eval_data["model_name"]
                for i, error in enumerate(eval_data.get("errors", [])):
                    question_idx = min(i, len(questions) - 1) 
                    
                    # Extract question text
                    if question_idx >= 0:
                        if isinstance(questions[question_idx], dict) and 'question' in questions[question_idx]:
                            question_text = questions[question_idx]['question']
                        elif hasattr(questions[question_idx], 'question'):
                            question_text = questions[question_idx].question
                        else:
                            question_text = str(questions[question_idx])
                    else:
                        question_text = "Unknown"
                        
                    writer.writerow([
                        model_name,
                        question_text,
                        error.get("error", "Unknown error")
                    ])
    
    # Display trait averages summary
    display_trait_averages(results_data, questions, logger)
    
    logger.info("Results saved successfully")


def display_trait_averages(results_data, questions, logger):
    """Display a summary of average scores by personality trait for each model in a table format."""
    logger.info("\n===== PERSONALITY TRAIT AVERAGES =====")
    
    # Get all unique traits from questions
    traits = set()
    for q in questions:
        if isinstance(q, dict) and 'trait' in q:
            traits.add(q['trait'])
        elif hasattr(q, 'trait'):
            traits.add(q.trait)
    
    traits = sorted(list(traits))
    
    # Calculate average scores by trait for each model
    model_trait_scores = {}
    for eval_data in results_data["model_evaluations"]:
        model_name = eval_data["model_name"]
        model_trait_scores[model_name] = {}
        
        # Group questions by trait
        trait_scores = {trait: [] for trait in traits}
        
        for i, response in enumerate(eval_data["responses"]):
            if i >= len(questions):
                continue
                
            question = questions[i]
            trait = None
            
            # Extract trait from question
            if isinstance(question, dict) and 'trait' in question:
                trait = question['trait']
            elif hasattr(question, 'trait'):
                trait = question.trait
                
            if trait and trait in trait_scores and response:
                score = response.get("score")
                if score:
                    trait_scores[trait].append(score)
        
        # Calculate average scores
        for trait in traits:
            scores = trait_scores[trait]
            if scores:
                avg_score = sum(scores) / len(scores)
                model_trait_scores[model_name][trait] = avg_score
            else:
                model_trait_scores[model_name][trait] = None
    
    # Display results in a table format
    model_names = list(model_trait_scores.keys())
    
    # Calculate column widths
    trait_width = max(len("Trait"), max(len(trait) for trait in traits))
    model_widths = [max(len(model), 8) for model in model_names]  # Min width of 8 for scores
    
    # Print header row
    header = f"| {'Trait'.ljust(trait_width)} |"
    for i, model in enumerate(model_names):
        header += f" {model.ljust(model_widths[i])} |"
    
    separator = f"+{'-' * (trait_width + 2)}+"
    for width in model_widths:
        separator += f"{'-' * (width + 2)}+"
    
    logger.info(separator)
    logger.info(header)
    logger.info(separator)
    
    # Print trait rows
    for trait in traits:
        row = f"| {trait.ljust(trait_width)} |"
        for i, model in enumerate(model_names):
            score = model_trait_scores[model].get(trait)
            if score is not None:
                score_str = f"{score:.2f}"
            else:
                score_str = "N/A"
            row += f" {score_str.ljust(model_widths[i])} |"
        logger.info(row)
    
    logger.info(separator)
    logger.info("\n=====================================")


def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f) 