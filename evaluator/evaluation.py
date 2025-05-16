#TODO: Load dataset using hugging face datasets , load_dataset ("benchmark_datasets/mtbench101.jsonl")

# TODO: Use the model "florence-2-large-ft"
# TODO: Get the predictions and write it into a jsonl file

# TODO: Do the evaluation using the code in cascade_evaluator.py and generic_evaluator.py
# OpenAI key is in the openai_config.yaml file

# TODO: Get the evaluation score for the predicions

import argparse
import json
import os
import yaml
import logging # Added for logging setup
import re # Add this import
from typing import Dict, Any # Added Dict, Any

import torch
from PIL import Image
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Import refactored evaluators directly
from generic_evaluator import GenericLLMEvaluator
from cascade_evaluator import CascadeEvaluator

# Removed OpenCompass specific imports for evaluators
# from opencompass.registry import ICL_EVALUATORS 

# If mmengine.ConfigDict is used for config, ensure it's available or use plain dicts.
# For consistency with evaluators, let's add the ConfigDict fallback.
try:
    from mmengine.config import ConfigDict
except ImportError:
    logging.warning("mmengine.config.ConfigDict not found for evaluation.py. Using a basic dict wrapper.")
    class ConfigDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a benchmark dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="benchmark_datasets/mtbench101.jsonl",
        help="Path to the benchmark dataset (JSONL format)."
    )
    parser.add_argument(
        "--output_predictions_file",
        type=str,
        default="predictions.jsonl",
        help="File to save the model predictions."
    )
    parser.add_argument(
        "--openai_config_path",
        type=str,
        default="openai_config.yaml",
        help="Path to OpenAI config YAML file (for evaluators)."
    )
    parser.add_argument(
        "--evaluator_type",
        type=str,
        default="GenericLLMEvaluator",
        choices=["GenericLLMEvaluator", "CascadeEvaluator"],
        help="Type of refactored evaluator to use."
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        default="gpt-4",
        help="Name of the judge model for LLM-based evaluation (e.g., gpt-3.5-turbo, gpt-4, gpt-4o)."
    )
    parser.add_argument(
        "--judge_system_prompt",
        type=str,
        default=None,
        help="System prompt content to be used by the LLM judge."
    )
    # Add arg for output directory for all evaluation files
    parser.add_argument(
        "--eval_output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save all evaluation output files (predictions, judge outputs, final scores)."
    )
    return parser.parse_args()

def load_openai_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    logger.warning(f"OpenAI config file not found at {config_path}")
    return {}


# Define a simple text postprocessor for the judge's output for TextSummarizer-like behavior
# This would be passed as `dict_postprocessor` to GenericLLMEvaluator
def simple_text_summarizer_postprocessor(judge_response_text: str) -> Dict[str, Any]:
    """Postprocessor to extract a score from judge response text using regex."""
    score = None
    lines = judge_response_text.strip().split('\n')

    # Pattern 1: Look for "Score: X", "Score: X/Y", or Chinese variants. Captures X.
    # Example: "Score: 4/10" -> captures "4"; "Score: 8.5" -> captures "8.5"
    score_keyword_pattern = r"(?:score|评[分价测]|得分)[:：]?\s*(\d+(?:\.\d+)?)(?:/\d+)?"

    # Pattern 2: Look for a standalone number, often at the end of a line/response.
    # This is a fallback and is more restrictive to avoid grabbing random numbers.
    # It checks if the number is likely a score (e.g., within 0-10 range if it's not explicitly labeled with 'Score:').
    # Looks for a number possibly at the end of a string, or on its own line.
    # Positive lookbehind for start of string or non-alphanumeric/non-dot/non-underscore, to avoid matching parts of words/numbers.
    # Positive lookahead for end of string or non-alphanumeric/non-dot/non-underscore.
    standalone_score_pattern = r"(?<![a-zA-Z0-9\._-])(\b\d+(?:\.\d+)?\b)(?![a-zA-Z0-9\._-])"

    for line in reversed(lines): # Check from last lines first
        line_cleaned = line.strip()
        if not line_cleaned: # Skip empty lines
            continue

        # Try Pattern 1: "Score: X" or "Score: X/Y"
        match = re.search(score_keyword_pattern, line_cleaned, re.IGNORECASE)
        if match:
            score_str = match.group(1) # The first capture group has the score value
            if score_str:
                try:
                    score = float(score_str)
                    break # Score found and successfully parsed
                except ValueError:
                    # Failed to convert, log or pass, then continue to next line or fallback
                    logger.warning(f"Found score-like text \"{score_str}\" with keyword but failed to parse as float.")
                    pass 
        
        if score is not None: # If score was found by keyword pattern, no need to check standalone
            break

        # Fallback: Try Pattern 2 (standalone number) only if keyword pattern didn't yield a score for this line
        # This is applied to the current line being processed.
        # We are looking for numbers that could be scores, especially if they are simple numbers on a line.

        # First, check if the entire line is just a number (strongest indicator for a standalone score)
        if re.fullmatch(r"\d+(?:\.\d+)?", line_cleaned):
            try:
                potential_score = float(line_cleaned)
                # For scores that are just numbers on a line, we might still want a range check
                # if there's a known, fixed score range (e.g. 1-5, 0-10). Assuming 0-10 for now as common.
                if 0 <= potential_score <= 10: # Adjust range if needed
                    score = potential_score
                    break # Found a score that is the entire line content and in range
            except ValueError:
                pass # Not a float, ignore

        if score is not None: # If score found as full line number, stop.
            break
            
        # If the line isn't *just* a number, try to find standalone numbers within it (e.g., at the end of a sentence)
        # using the standalone_score_pattern. We take the *last* such number on the line.
        # This is more speculative, so we apply a stricter range check (e.g. 0-10).
        else:
            all_standalone_matches = list(re.finditer(standalone_score_pattern, line_cleaned))
            if all_standalone_matches:
                last_match_str = all_standalone_matches[-1].group(1) # Get the string of the last matched number
                try:
                    potential_score = float(last_match_str)
                    # Apply a range check for these less certain scores
                    if 0 <= potential_score <= 10: # Common score range
                        score = potential_score
                        # If we found a standalone number that fits criteria, break from line loop
                        break 
                except ValueError:
                    pass # Not a float, ignore

        if score is not None: # If score found by any means on this line, break from line loop
            break
            
    return {"score": score, "raw_judge_response": judge_response_text}


def main():
    args = parse_args()
    os.makedirs(args.eval_output_dir, exist_ok=True)
    # Adjust output_predictions_file to be inside eval_output_dir
    output_predictions_filepath = os.path.join(args.eval_output_dir, os.path.basename(args.output_predictions_file))


    logger.info("**************************************************************************************")
    logger.info("** IMPORTANT NOTE: The specified model 'microsoft/Florence-2-large-ft' is a       **")
    logger.info("** vision-language model, primarily designed for tasks involving images.          **")
    logger.info("** The dataset 'mtbench101.jsonl' appears to be a text-only conversational        **")
    logger.info("** benchmark. Using Florence-2 on this dataset may not yield meaningful results   **")
    logger.info("** or may require specific text-only prompting strategies not inherent to the model.**")
    logger.info("**************************************************************************************")

    # 1. Load dataset
    args.dataset_path = "benchmark_datasets/mtbench101.jsonl"
    n = 5  # Number of lines to load for testing
    logger.info(f"Loading dataset from {args.dataset_path} (first {n} lines)...")
    try:
        raw_dataset_list = []
        with open(args.dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                raw_dataset_list.append(json.loads(line))
        logger.info(f"Dataset loaded. Number of conversations: {len(raw_dataset_list)} (limited to {n})")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        return

    # 2. Load Model and Tokenizer
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    logger.info(f"Loading model: {model_name_or_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        model = model.eval()
        logger.info("Model and tokenizer loaded.")
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}", exc_info=True)
        return

    # 3. Get Predictions
    logger.info(f"Generating predictions and saving to {output_predictions_filepath}...")
    predictions_data = []

    for conversation_data in tqdm(raw_dataset_list, desc="Generating Predictions"):
        conv_id = conversation_data.get("id", "unknown_id")
        history = conversation_data.get("history", [])
        task_category = conversation_data.get("task", "unknown_task")
        
        if not history or not isinstance(history, list) or not history[-1].get("user"):
            predictions_data.append({
                "id": conv_id, "task_category": task_category,
                "prediction": "Error: Invalid or empty history, or last turn not from user.",
                "reference_answer": "N/A", "full_history": history
            })
            continue

        current_prompt_text = history[-1]["user"]
        reference_answer = history[-1].get("bot", "N/A")
        model_input_text = current_prompt_text

        try:
            inputs = tokenizer(model_input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True
                )
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            parsed_answer = result
            predictions_data.append({
                "id": conv_id, "task_category": task_category,
                "model_input": model_input_text, "prediction": parsed_answer,
                "reference_answer": reference_answer, "full_history": history
            })
        except Exception as e:
            logger.warning(f"Error generating prediction for conversation ID {conv_id}: {e}")
            predictions_data.append({
                "id": conv_id, "task_category": task_category,
                "model_input": model_input_text, "prediction": f"Error: {e}",
                "reference_answer": reference_answer, "full_history": history
            })

    with open(output_predictions_filepath, 'w') as f:
        for item in predictions_data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Predictions saved to {output_predictions_filepath}")

    # Prepare Hugging Face Dataset for the evaluator from predictions_data
    dataset_for_eval_dict = {
        "query": [], "prediction": [], "reference": [], "id": [], "task_category": []
    }
    for item in predictions_data:
        # Only include items that didn't error out during prediction generation
        if not item["prediction"].startswith("Error:"):
            dataset_for_eval_dict["query"].append(item["model_input"])
            dataset_for_eval_dict["prediction"].append(item["prediction"])
            ref = item["reference_answer"]
            # Ensure reference is a string, not a list (common in some datasets)
            if isinstance(ref, list):
                ref = " ".join(r for r in ref if isinstance(r, str)) if all(isinstance(r, str) for r in ref) else str(ref)
            dataset_for_eval_dict["reference"].append(ref)
            dataset_for_eval_dict["id"].append(item["id"])
            dataset_for_eval_dict["task_category"].append(item["task_category"])

    if not dataset_for_eval_dict["prediction"]:
        logger.warning("No successful predictions to evaluate after filtering. Exiting.")
        # Save score file indicating no successful predictions
        score_file_name = f"{model_name_or_path.replace('/', '_')}_score.txt"
        score_file_path = os.path.join(args.eval_output_dir, score_file_name)
        try:
            with open(score_file_path, 'w') as f:
                f.write(f"Model: {model_name_or_path}\n")
                f.write(f"Dataset: {args.dataset_path}\n")
                f.write(f"Evaluator: {args.evaluator_type}\n")
                f.write(f"Judge Model: {args.judge_model_name}\n")
                f.write(f"Final Score: No successful predictions to evaluate\n")
            logger.info(f"Evaluation score file (indicating no predictions) saved to {score_file_path}")
        except Exception as e:
            logger.error(f"Failed to write score file: {e}", exc_info=True)
        return

    eval_hf_dataset = Dataset.from_dict(dataset_for_eval_dict)

    # 4. Prepare Evaluator Configuration
    logger.info(f"Configuring evaluator: {args.evaluator_type} with judge model: {args.judge_model_name}...")
    openai_params = load_openai_config(args.openai_config_path)

    # Base judge_cfg, common for both evaluator types
    judge_cfg_dict = {
        "model": args.judge_model_name,  # path/model key for GenericLLMEvaluator
        "key": openai_params.get("api_key"), # API key from YAML
        "openai_api_base": openai_params.get("base_url"), # base_url from YAML (if present)
        "temperature": 0.0,
        "max_out_len": 1024, # Increased for potentially more detailed judge responses
        "query_per_second": 1, # Default QPS
        "system_prompt_content": args.judge_system_prompt # Pass the system prompt
    }
    
    # Specific prompt template for the judge.
    # This is a generic one, might need adjustment based on task.
    prompt_template_dict = {
        "template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. "
                     "Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. "
                     "Begin your evaluation by providing a short explanation. Be as objective as possible. "
                     "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly outputting a single line with only the score. "
                     "Do not output any other text after the score. "
                     "\n\n[Question]\n{query}\n\n[The Start of Assistant's Answer]\n{prediction}\n[The End of Assistant's Answer]"
                     "\n\n[Reference Answer (if available)]\n{reference}\n[The End of Reference Answer]",
        "input_columns": ["query", "prediction", "reference"], # Ensure these align with dataset/what's needed
    }

    evaluator_output_path = os.path.join(args.eval_output_dir, f"{args.evaluator_type}_results.json")

    # Convert to ConfigDict if mmengine is available and was imported successfully
    # otherwise, they remain plain dicts which GenericLLMEvaluator can also handle.
    try:
        ConfigDict # Check if ConfigDict is defined (either imported or the dummy)
        judge_config = ConfigDict(judge_cfg_dict)
        prompt_template_config = ConfigDict(prompt_template_dict)
    except NameError: # Should not happen if the try-except for mmengine import at top worked
        logger.warning("ConfigDict class not found, using plain dicts for evaluator config.")
        judge_config = judge_cfg_dict
        prompt_template_config = prompt_template_dict


    # 5. Initialize and Run Evaluator
    eval_results = {}
    if args.evaluator_type == "GenericLLMEvaluator":
        evaluator = GenericLLMEvaluator(
            judge_cfg=judge_config,
            prompt_template=prompt_template_config,
            dict_postprocessor=simple_text_summarizer_postprocessor, # Example postprocessor
            output_path=evaluator_output_path
        )
    elif args.evaluator_type == "CascadeEvaluator":
        # CascadeEvaluator needs an llm_evaluator_cfg, which is composed of judge_cfg and prompt_template
        llm_eval_config_for_cascade = ConfigDict({
            "judge_cfg": judge_config,
            "prompt_template": prompt_template_config,
            "dict_postprocessor": simple_text_summarizer_postprocessor,
            # output_path for internal LLM judge is handled by CascadeEvaluator
        })
        evaluator = CascadeEvaluator(
            llm_evaluator_cfg=llm_eval_config_for_cascade,
            # rule_evaluator_cfg=None, # Add config if using a rule_evaluator_class
            # sample_score_fn=None, # Or provide a sample_score_fn
            output_path=evaluator_output_path
        )
    else:
        logger.error(f"Unsupported evaluator type: {args.evaluator_type}")
        return

    final_score_value = "N/A"
    if evaluator:
        try:
            logger.info(f"Running evaluation with {args.evaluator_type}...")
            evaluation_results = evaluator.score(
                predictions=list(eval_hf_dataset["prediction"]),
                test_set=eval_hf_dataset # Pass the full dataset with query, reference etc.
            )
            
            logger.info("\nRaw Evaluation Results:")
            # Use a simple way to pretty print if json.dumps causes issues with complex objects
            try:
                logger.info(json.dumps(evaluation_results, indent=4))
            except TypeError:
                logger.info(str(evaluation_results))

            # Extract final score for saving
            if isinstance(evaluation_results, dict):
                if "average_score" in evaluation_results: # From GenericLLMEvaluator direct output
                    final_score_value = evaluation_results["average_score"]
                    logger.info(f"\nAverage Judge Score: {final_score_value:.2f} (Scored items: {evaluation_results.get('num_scored', 'N/A')})")
                elif "accuracy" in evaluation_results: # From CascadeEvaluator direct output
                    final_score_value = evaluation_results["accuracy"]
                    logger.info(f"\nOverall Accuracy: {final_score_value:.2f}")
                else:
                    logger.warning("Could not determine a single summary score from evaluation results.")
            else:
                logger.warning("Evaluation results format not a dictionary as expected.")

        except Exception as e:
            logger.error(f"Error during evaluation with refactored evaluators: {e}", exc_info=True)
    
    # Write the score to a text file
    score_file_name = f"{model_name_or_path.replace('/', '_')}_score.txt"
    score_file_path = os.path.join(args.eval_output_dir, score_file_name)
    try:
        with open(score_file_path, 'w') as f:
            f.write(f"Model: {model_name_or_path}\n")
            f.write(f"Dataset: {args.dataset_path}\n")
            f.write(f"Evaluator: {args.evaluator_type}\n")
            f.write(f"Judge Model: {args.judge_model_name}\n")
            f.write(f"Final Score: {final_score_value}\n")
        logger.info(f"Evaluation score saved to {score_file_path}")
    except Exception as e:
        logger.error(f"Failed to write score to file {score_file_path}: {e}", exc_info=True)

    logger.info("\nEvaluation script finished.")

if __name__ == "__main__":
    main()