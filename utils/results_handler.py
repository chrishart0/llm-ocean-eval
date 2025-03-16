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
        
        # Write header with Trait column and Reverse flag
        header = ["Question", "Trait", "Reverse"] + model_names
        writer.writerow(header)
        
        # Write scores for each question
        for i, question in enumerate(questions):
            # Extract question text, trait, and reverse flag from different possible formats
            if isinstance(question, dict):
                question_text = question.get('question', str(question))
                trait = question.get('trait', 'Unknown').strip()
                is_reverse = "Yes" if question.get('reverse', False) else "No"
            elif hasattr(question, 'question'):
                question_text = question.question
                trait = question.trait.strip() if hasattr(question, 'trait') else 'Unknown'
                is_reverse = "Yes" if (hasattr(question, 'reverse') and question.reverse) else "No"
            else:
                question_text = str(question)
                trait = 'Unknown'
                is_reverse = "No"
                
            row = [question_text, trait, is_reverse]
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
    
    # Also save a trait-averaged CSV for easier analysis
    trait_csv_filename = f"results/trait_averages_{timestamp}.csv"
    logger.info(f"Saving trait averages to {trait_csv_filename}")
    
    # Get all unique traits
    traits = set()
    for q in questions:
        if isinstance(q, dict) and 'trait' in q:
            traits.add(q['trait'].strip())
        elif hasattr(q, 'trait'):
            traits.add(q.trait.strip())
    
    traits = sorted(list(traits))
    
    # Calculate trait averages for each model
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
            is_reverse = False
            
            # Extract trait and reverse flag from question
            if isinstance(question, dict):
                if 'trait' in question:
                    trait = question['trait'].strip()
                is_reverse = question.get('reverse', False)
            elif hasattr(question, 'trait'):
                trait = question.trait.strip()
                is_reverse = hasattr(question, 'reverse') and question.reverse
                
            if trait and trait in trait_scores and response:
                score = response.get("score")
                if score:
                    # Reverse the score if needed (1→5, 2→4, 3→3, 4→2, 5→1)
                    if is_reverse:
                        score = 6 - score
                    trait_scores[trait].append(score)
        
        # Calculate average scores
        for trait in traits:
            scores = trait_scores[trait]
            if scores:
                avg_score = sum(scores) / len(scores)
                model_trait_scores[model_name][trait] = avg_score
            else:
                model_trait_scores[model_name][trait] = None
    
    # Write trait averages to CSV
    with open(trait_csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["Trait"] + model_names
        writer.writerow(header)
        
        # Write trait averages
        for trait in traits:
            row = [trait]
            for model_name in model_names:
                score = model_trait_scores[model_name].get(trait)
                if score is not None:
                    row.append(f"{score:.2f}")
                else:
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
            writer.writerow(["Model", "Question", "Trait", "Reverse", "Error"])
            
            # Write errors
            for eval_data in results_data["model_evaluations"]:
                model_name = eval_data["model_name"]
                for i, error in enumerate(eval_data.get("errors", [])):
                    question_idx = min(i, len(questions) - 1) 
                    
                    # Extract question text, trait, and reverse flag
                    if question_idx >= 0:
                        question = questions[question_idx]
                        if isinstance(question, dict):
                            question_text = question.get('question', str(question))
                            trait = question.get('trait', 'Unknown').strip()
                            is_reverse = "Yes" if question.get('reverse', False) else "No"
                        elif hasattr(question, 'question'):
                            question_text = question.question
                            trait = question.trait.strip() if hasattr(question, 'trait') else 'Unknown'
                            is_reverse = "Yes" if (hasattr(question, 'reverse') and question.reverse) else "No"
                        else:
                            question_text = str(question)
                            trait = 'Unknown'
                            is_reverse = "No"
                    else:
                        question_text = "Unknown"
                        trait = "Unknown"
                        is_reverse = "No"
                        
                    writer.writerow([
                        model_name,
                        question_text,
                        trait,
                        is_reverse,
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
            traits.add(q['trait'].strip())
        elif hasattr(q, 'trait'):
            traits.add(q.trait.strip())
    
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
            is_reverse = False
            
            # Extract trait and reverse flag from question
            if isinstance(question, dict):
                if 'trait' in question:
                    trait = question['trait'].strip()
                is_reverse = question.get('reverse', False)
            elif hasattr(question, 'trait'):
                trait = question.trait.strip()
                is_reverse = hasattr(question, 'reverse') and question.reverse
                
            if trait and trait in trait_scores and response:
                score = response.get("score")
                if score:
                    # Reverse the score if needed (1→5, 2→4, 3→3, 4→2, 5→1)
                    if is_reverse:
                        score = 6 - score
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