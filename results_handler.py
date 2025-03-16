"""Handle saving and loading evaluation results."""

import os
import json
from datetime import datetime
from pathlib import Path


def save_results(results, questions, model_versions, logger):
    """Save evaluation results to JSON file."""
    logger.info("Saving results")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation_{timestamp}.json"
    
    # Save results to JSON file
    with open(results_file, "w") as f:
        json.dump({
            "questions": questions,
            "model_versions": model_versions,
            "model_evaluations": results["model_evaluations"]
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    return results_file


def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f) 