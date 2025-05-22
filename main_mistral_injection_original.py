import argparse
import json
import os
import yaml
import re
from typing import Dict, Any, List, Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Assuming evaluator scripts are in a subdirectory or accessible in PYTHONPATH
# If they are in 'evaluator' subdirectory:
from evaluator.generic_evaluator import GenericLLMEvaluator
# from evaluator.cascade_evaluator import CascadeEvaluator # If needed

# Import for Adapter
from adapter.mistral_adapter import DCTAdapter # Assuming DCTAdapter is in adapter/my_adapter.py
from runner.train import train_model, freeze_model_except_adapters

try:
    from mmengine.config import ConfigDict
except ImportError:
    print("mmengine.config.ConfigDict not found. Using a basic dict wrapper.")
    class ConfigDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value

# Setup basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# --- Adapter Helper Functions (from main_chartQA.py) ---
def get_parent_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    names = name.split('.')
    parent = model
    for n in names[:-1]:
        parent = getattr(parent, n)
    return parent

def inject_adapters(
    model: torch.nn.Module,
    adapter_cls: type,
    base_adapter_args: dict, # Renamed from adapter_args for clarity
    layers_config: List[Dict[str, str]] # Expected format: [{'name': 'layer_name_pattern_to_match'}]
) -> torch.nn.Module:
    print(f"Starting adapter injection with {adapter_cls.__name__}...")
    for name, module in model.named_modules(): # Iterate over all module names in the model
        for layer_conf in layers_config:
            # Check if the current module's name matches the configuration name
            # The original code used 'in', which allows pattern matching.
            # If exact names are always provided in layer_conf, '==' could be used.
            # Sticking to 'in' to maintain original flexibility if patterns are used.
            if layer_conf['name'] in name:
                # Ensure we are matching the exact module intended, not a submodule containing the name
                # This check assumes layer_conf['name'] is the full name of the target module
                if name == layer_conf['name']:
                    print(f"Matched target layer for injection: {name}")
                    try:
                        parent = get_parent_module(model, name)
                        original_module = getattr(parent, name.split('.')[-1])
                        
                        current_adapter_args = base_adapter_args.copy()

                        if hasattr(original_module, 'out_features') and isinstance(getattr(original_module, 'out_features'), int):
                            actual_in_features = original_module.out_features
                            print(f"Dynamically setting adapter 'in_features' for {name} to {actual_in_features} (derived from original_module.out_features).")
                            current_adapter_args['in_features'] = actual_in_features
                        else:
                            print(f"Original module {name} (type: {type(original_module)}) does not have an integer 'out_features' attribute. "
                                           f"Using 'in_features' from base_adapter_args: {current_adapter_args.get('in_features')}. "
                                           f"This might lead to errors if incorrect for this layer.")

                        adapter_instance = adapter_cls(**current_adapter_args)
                        setattr(parent, name.split('.')[-1], torch.nn.Sequential(original_module, adapter_instance))
                        print(f"Successfully injected adapter after {name} with args: {current_adapter_args}")
                    except Exception as e:
                        print(f"Failed to inject adapter into {name}: {e}") # Consider adding exc_info=True if Traceback is needed
    return model

# --- Placeholder for functions from runner.train (implement or import them) ---
from runner.train import freeze_model_except_adapters, train_model

# --- Helper Functions (adapted from evaluation.py) ---

def load_openai_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    print(f"OpenAI config file not found at {config_path}")
    return {}

def simple_text_summarizer_postprocessor(judge_response_text: str) -> Dict[str, Any]:
    """Postprocessor to extract a score from judge response text using regex."""
    score = None
    lines = judge_response_text.strip().split('\n')
    score_keyword_pattern = r"(?:score|评[分价测]|得分)[:：]?\s*(\d+(?:\.\d+)?)(?:/\d+)?"
    standalone_score_pattern = r"(?<![a-zA-Z0-9\._-])(\b\d+(?:\.\d+)?\b)(?![a-zA-Z0-9\._-])"

    for line in reversed(lines):
        line_cleaned = line.strip()
        if not line_cleaned:
            continue
        match = re.search(score_keyword_pattern, line_cleaned, re.IGNORECASE)
        if match:
            score_str = match.group(1)
            if score_str:
                try:
                    score = float(score_str)
                    break
                except ValueError:
                    print(f"Found score-like text \"{score_str}\" with keyword but failed to parse as float.")
                    pass
        if score is not None:
            break
        if re.fullmatch(r"\d+(?:\.\d+)?", line_cleaned):
            try:
                potential_score = float(line_cleaned)
                if 0 <= potential_score <= 10:
                    score = potential_score
                    break
            except ValueError:
                pass
        if score is not None:
            break
        else:
            all_standalone_matches = list(re.finditer(standalone_score_pattern, line_cleaned))
            if all_standalone_matches:
                last_match_str = all_standalone_matches[-1].group(1)
                try:
                    potential_score = float(last_match_str)
                    if 0 <= potential_score <= 10:
                        score = potential_score
                        break
                except ValueError:
                    pass
        if score is not None:
            break
    return {"score": score, "raw_judge_response": judge_response_text}

def generate_predictions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_split: Dataset,
    device: str,
    max_new_tokens: int = 512
) -> List[Dict[str, Any]]:
    print(f"Generating predictions for {len(dataset_split)} samples...")
    predictions_data = []
    for example in tqdm(dataset_split, desc="Generating Predictions"):
        conv_id = example.get("id", "unknown_id")
        # Assuming dataset has 'query' and 'reference' (optional)
        # For mtbench, 'history' is used. We need the last user turn as query.
        history = example.get("history")
        if not history or not isinstance(history, list) or not history[-1].get("user"):
            current_prompt_text = example.get("query", "") # Fallback if history is not as expected
            if not current_prompt_text:
                 print(f"Skipping item {conv_id} due to missing user prompt in history or query field.")
                 predictions_data.append({
                    "id": conv_id, "task_category": example.get("task_category", "N/A"),
                    "model_input": "Error: Missing prompt", "prediction": "Error: Missing prompt",
                    "reference_answer": example.get("reference", "N/A"), "full_history": history
                 })
                 continue
        else:
            current_prompt_text = history[-1]["user"]

        reference_answer = history[-1].get("bot", example.get("reference", "N/A"))
        
        model_input_text = current_prompt_text

        try:
            inputs = tokenizer(model_input_text, return_tensors="pt", truncation=True, max_length=2048).to(device) # Added truncation
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True, # Consistent with evaluation.py
                    pad_token_id=tokenizer.eos_token_id # Add pad_token_id for open-ended generation
                )
            # Ensure decoding handles cases where input is part of the output
            # For instruct models, often the prompt is not repeated.
            # If prompt is repeated, use: result = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # A simple way to remove prompt if it's there, more robust methods might be needed
            if result.startswith(model_input_text):
                parsed_answer = result[len(model_input_text):].strip()
            else:
                parsed_answer = result.strip()

            predictions_data.append({
                "id": conv_id,
                "task_category": example.get("task_category", "N/A"),
                "model_input": model_input_text,
                "prediction": parsed_answer,
                "reference_answer": reference_answer,
                "full_history": history
            })
        except Exception as e:
            print(f"Error generating prediction for ID {conv_id}: {e}")
            predictions_data.append({
                "id": conv_id, "task_category": example.get("task_category", "N/A"),
                "model_input": model_input_text, "prediction": f"Error: {e}",
                "reference_answer": reference_answer, "full_history": history
            })
    return predictions_data

def run_evaluation_pipeline(
    model_name_or_path: str,
    model_to_evaluate: AutoModelForCausalLM, # Pass the loaded model
    tokenizer: AutoTokenizer, # Pass the loaded tokenizer
    eval_dataset: Dataset,
    args: argparse.Namespace,
    output_suffix: str = ""
) -> Optional[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_to_evaluate.to(device)
    model_to_evaluate.eval()

    # 1. Generate Predictions
    predictions_output_filename = f"predictions_{output_suffix}.jsonl"
    predictions_output_filepath = os.path.join(args.eval_output_dir, predictions_output_filename)
    
    generated_preds_list = generate_predictions(model_to_evaluate, tokenizer, eval_dataset, device, args.max_new_tokens)
    
    with open(predictions_output_filepath, 'w') as f:
        for item in generated_preds_list:
            f.write(json.dumps(item) + '\n')
    print(f"Predictions for {output_suffix} saved to {predictions_output_filepath}")

    # 2. Prepare dataset for evaluator
    dataset_for_eval_dict = {
        "query": [], "prediction": [], "reference": [], "id": [], "task_category": []
    }
    valid_predictions_count = 0
    for item in generated_preds_list:
        if not item["prediction"].startswith("Error:"):
            dataset_for_eval_dict["query"].append(item["model_input"])
            dataset_for_eval_dict["prediction"].append(item["prediction"])
            ref = item["reference_answer"]
            if isinstance(ref, list): # Ensure reference is a string
                ref = " ".join(r for r in ref if isinstance(r, str)) if all(isinstance(r, str) for r in ref) else str(ref)
            dataset_for_eval_dict["reference"].append(ref)
            dataset_for_eval_dict["id"].append(item["id"])
            dataset_for_eval_dict["task_category"].append(item.get("task_category", "N/A"))
            valid_predictions_count += 1
    
    if valid_predictions_count == 0:
        print(f"No successful predictions to evaluate for {output_suffix}. Skipping evaluation.")
        final_score_value = "N/A (No valid predictions)"
    else:
        eval_hf_dataset = Dataset.from_dict(dataset_for_eval_dict)
        print(f"Prepared {len(eval_hf_dataset)} samples for the evaluator for {output_suffix}.")

        # 3. Configure and Run Evaluator
        openai_params = load_openai_config(args.openai_config_path)
        judge_cfg_dict = {
            "model": args.judge_model_name,
            "key": openai_params.get("api_key"),
            "openai_api_base": openai_params.get("base_url"),
            "temperature": 0.0, "max_out_len": 1024, "query_per_second": 1,
            "system_prompt_content": args.judge_system_prompt
        }
        prompt_template_dict = {
            "template": "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. "
                         "Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. "
                         "Begin your evaluation by providing a short explanation. Be as objective as possible. "
                         "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly outputting a single line with only the score. "
                         "Do not output any other text after the score. "
                         "\n\n[Question]\n{query}\n\n[The Start of Assistant's Answer]\n{prediction}\n[The End of Assistant's Answer]"
                         "\n\n[Reference Answer (if available)]\n{reference}\n[The End of Reference Answer]",
            "input_columns": ["query", "prediction", "reference"],
        }
        evaluator_results_filename = f"{args.evaluator_type}_results_{output_suffix}.json"
        evaluator_output_path = os.path.join(args.eval_output_dir, evaluator_results_filename)

        judge_config = ConfigDict(judge_cfg_dict)
        prompt_template_config = ConfigDict(prompt_template_dict)

        evaluator = GenericLLMEvaluator(
            judge_cfg=judge_config,
            prompt_template=prompt_template_config,
            dict_postprocessor=simple_text_summarizer_postprocessor,
            output_path=evaluator_output_path
        )

        print(f"Running evaluation with {args.evaluator_type} for {output_suffix}...")
        evaluation_results = evaluator.score(
            predictions=list(eval_hf_dataset["prediction"]),
            test_set=eval_hf_dataset
        )
        print(f"Raw Evaluation Results for {output_suffix}:")
        try:
            print(json.dumps(evaluation_results, indent=4))
        except TypeError:
            print(str(evaluation_results))

        final_score_value = "N/A"
        if isinstance(evaluation_results, dict) and "average_score" in evaluation_results:
            final_score_value = evaluation_results["average_score"]
            num_scored = evaluation_results.get('num_scored', 'N/A')
            print(f"Average Judge Score for {output_suffix}: {final_score_value:.2f} (Scored items: {num_scored})")
        else:
            print(f"Could not determine average_score from evaluation results for {output_suffix}.")


    # 4. Save Score File
    score_file_name = f"{model_name_or_path.replace('/', '_')}_{output_suffix}_score.txt"
    score_file_path = os.path.join(args.eval_output_dir, score_file_name)
    try:
        with open(score_file_path, 'w') as f:
            f.write(f"Model: {model_name_or_path} ({output_suffix})\n")
            f.write(f"Dataset: {args.dataset_path}\n")
            f.write(f"Evaluator: {args.evaluator_type}\n")
            f.write(f"Judge Model: {args.judge_model_name}\n")
            f.write(f"Evaluation Split Size: {len(eval_dataset) if eval_dataset is not None else 'N/A'}\n")
            f.write(f"Final Score: {final_score_value}\n")
        print(f"Evaluation score for {output_suffix} saved to {score_file_path}")
    except Exception as e:
        print(f"Failed to write score to file {score_file_path}: {e}") # Consider adding exc_info=True if Traceback is needed
    
    if isinstance(final_score_value, (float, int)):
        return float(final_score_value)
    return None

# --- Main Script Logic ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate original and adapted Mistral models.")
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Path to the base Mistral model.")
    parser.add_argument("--dataset_path", type=str, default="evaluator/benchmark_datasets/mtbench101.jsonl", help="Path to the benchmark dataset (JSONL format).")
    parser.add_argument("--num_train_samples", type=int, default=50, help="Number of samples for training/adaptation.")
    parser.add_argument("--num_eval_samples", type=int, default=50, help="Number of samples for evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation.")
    parser.add_argument("--eval_output_dir", type=str, default="./eval_results_injection", help="Directory for all outputs.")
    parser.add_argument("--openai_config_path", type=str, default="evaluator/openai_config.yaml", help="Path to OpenAI config.")
    parser.add_argument("--evaluator_type", type=str, default="GenericLLMEvaluator", choices=["GenericLLMEvaluator"], help="Evaluator type.")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4", help="Judge model name.")
    parser.add_argument("--judge_system_prompt", type=str, default=None, help="System prompt for the judge.")

    # Arguments for Adapter Injection
    parser.add_argument("--do_adapter_injection", action='store_true', help="Enable adapter injection.")
    parser.add_argument("--adapter_layers_json", type=str, default="""[{"name": "model.layers.15.mlp.gate_proj"}, {"name": "model.layers.15.mlp.up_proj"}]""",
                        help="""JSON string of layer configurations. Each dict must have a "name" key specifying the layer name/pattern. E.g., '[{"name": "block.1"}, {"name": "mlp"}]'.""")
    parser.add_argument(
        "--adapter_params_json",
        type=str,
        default='{"in_features": 4096, "reduction_factor": 16}',
        help="""JSON string of parameters for DCTAdapter. E.g., {"in_features": 4096, "reduction_factor": 16}. This depends on your DCTAdapter's __init__ method."""
    )
    parser.add_argument("--perform_adapter_training", action='store_true', help="Enable training after adapter injection.")
    parser.add_argument("--adapter_lr", type=float, default=1e-4, help="Learning rate for adapter training.")
    parser.add_argument("--adapter_epochs", type=int, default=1, help="Number of epochs for adapter training.")

    args = parser.parse_args()

    os.makedirs(args.eval_output_dir, exist_ok=True)

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Load and Split Dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        # Load the full dataset
        full_dataset_list = []
        with open(args.dataset_path, 'r') as f:
            for line in f:
                full_dataset_list.append(json.loads(line))
        
        if len(full_dataset_list) < args.num_train_samples + args.num_eval_samples:
            print(f"Dataset has {len(full_dataset_list)} samples, but "
                         f"{args.num_train_samples + args.num_eval_samples} are required for train+eval. Aborting.")
            return

        # Split dataset
        train_list = full_dataset_list[:args.num_train_samples]
        eval_list = full_dataset_list[args.num_train_samples : args.num_train_samples + args.num_eval_samples]

        # Convert to Hugging Face Dataset objects
        # Need to ensure the lists of dicts are correctly formatted for Dataset.from_list
        # Assuming each item in train_list/eval_list is a dict compatible with Dataset.from_list
        train_dataset = Dataset.from_list(train_list)
        eval_dataset = Dataset.from_list(eval_list)
        
        print(f"Dataset loaded. Training samples: {len(train_dataset)}, Evaluation samples: {len(eval_dataset)}")

    except Exception as e:
        print(f"Error loading or splitting dataset: {e}") # Consider adding exc_info=True if Traceback is needed
        return

    # 2. Load Tokenizer (shared for both models)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Set pad token if not present
            print(f"Tokenizer pad_token was None, set to eos_token: {tokenizer.eos_token}")
    except Exception as e:
        print(f"Failed to load tokenizer for {args.model_name_or_path}: {e}") # Consider adding exc_info=True if Traceback is needed
        return
        
    # --- Evaluate Original Model ---
    print(f"--- Starting Evaluation for Original Model: {args.model_name_or_path} ---")
    try:
        original_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        run_evaluation_pipeline(
            model_name_or_path=args.model_name_or_path,
            model_to_evaluate=original_model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            args=args,
            output_suffix="original"
        )
        del original_model # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during original model evaluation: {e}") # Consider adding exc_info=True if Traceback is needed


    # --- Adapter Injection and Evaluation for Adapted Model ---
    print(f"--- Starting Evaluation for Adapted Model: {args.model_name_or_path} + Adapter ---")
    adapted_model = None 
    model_for_adapted_eval_name = args.model_name_or_path 
    # Default to base model name
    args.do_adapter_injection = True
    try:
        if args.do_adapter_injection:
            print(f"Loading base model ({args.model_name_or_path}) for adapter injection...")
            adapted_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            model_for_adapted_eval_name = f"{args.model_name_or_path}_adapted"

            model_arch_path = os.path.join(args.eval_output_dir, "model_archi.txt")
            try:
                with open(model_arch_path, 'w') as f:
                    f.write(str(adapted_model))
                print(f"Model architecture written to {model_arch_path}")
            except Exception as e:
                print(f"Failed to write model architecture: {e}")

            print("Hello")
            try: 
                adapted_model = inject_adapters(adapted_model, DCTAdapter, config['adapter']['params'], config['adapter']['layers'])
                print("Adapter injection process finished.")
                freeze_model_except_adapters(adapted_model)
                print("ADAPTER MODEL ARCHITECTURE")
                print(adapted_model)

            except Exception as e:
                print(e)
            

            
            
            if args.perform_adapter_training:
                if train_dataset is None or len(train_dataset) == 0:
                    print("Adapter training requested, but train_dataset is empty or None. Skipping training.")
                else:
                    print(f"Performing adapter training using {len(train_dataset)} samples...")
                    # Ensure model is on the correct device for training
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    adapted_model.to(device)
                    adapted_model = train_model_with_adapter(adapted_model, tokenizer, train_dataset, args) # Placeholder call
                    print("Adapter training finished.")
            else:
                print("Adapter training not requested (perform_adapter_training=False).")
        else:
            print("Adapter injection not requested (do_adapter_injection=False). Evaluating base model as 'adapted' model.")
            # adapted_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            # model_for_adapted_eval_name = f"{args.model_name_or_path}_base_as_adapted" 

        if adapted_model: 
            print("Evaluation of adapted model--------------------")
            run_evaluation_pipeline(
                model_name_or_path=model_for_adapted_eval_name, 
                model_to_evaluate=adapted_model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                args=args,
                output_suffix="adapted" 
            )
        else:
            print("Adapted model was not loaded or created. Skipping evaluation for adapted model.")

        del adapted_model 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error during adapted model setup or evaluation: {e}") # Consider adding exc_info=True if Traceback is needed

    print("--- Main Mistral Injection Script Finished ---")

if __name__ == "__main__":
    main()