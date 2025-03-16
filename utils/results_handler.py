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
                    score = eval_data["responses"][i]["score"]
                    row.append(score)
                except (IndexError, KeyError):
                    row.append("N/A")
            writer.writerow(row)
    
    logger.info("Results saved successfully")


def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f) 