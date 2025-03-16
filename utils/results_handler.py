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
            row = [question]
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
                    writer.writerow([
                        model_name,
                        questions[question_idx] if question_idx >= 0 else "Unknown",
                        error.get("error", "Unknown error")
                    ])
    
    logger.info("Results saved successfully")


def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f) 