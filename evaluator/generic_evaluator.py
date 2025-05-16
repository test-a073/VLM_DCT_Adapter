import os
import json
import logging
import time
from typing import Dict, List, Optional, Any

# Attempt to import openai, require manual installation if not found
try:
    import openai
except ImportError:
    raise ImportError("The 'openai' library is required for GenericLLMEvaluator. Please install it by running 'pip install openai'.")

from datasets import Dataset
# from mmengine.config import ConfigDict # Can be replaced by standard dict if preferred, but not strictly an OC dependency. Keeping for now for compatibility with existing configs.
# For simplicity, we'll assume ConfigDict is available via mmengine or user installs it. If not, evaluation.py needs to pass plain dicts.
try:
    from mmengine.config import ConfigDict
except ImportError:
    # Define a dummy ConfigDict if mmengine is not available, to avoid crashing.
    # User should install mmengine for full compatibility if using ConfigDicts in their configs.
    logging.warning("mmengine.config.ConfigDict not found. Using a basic dict wrapper. Consider installing mmengine.")
    class ConfigDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value


# Removed OpenCompass specific imports:
# from opencompass.openicl.icl_evaluator import BaseEvaluator
# from opencompass.openicl.icl_inferencer import GenInferencer
# from opencompass.openicl.icl_retriever import ZeroRetriever
# from opencompass.registry import (DICT_POSTPROCESSORS, ICL_PROMPT_TEMPLATES, TEXT_POSTPROCESSORS)
# from opencompass.utils import build_dataset_from_cfg, build_model_from_cfg


class GenericLLMEvaluator: # Removed inheritance from BaseEvaluator
    """Generic LLM evaluator - Refactored to be OpenCompass independent."""

    def __init__(
        self,
        judge_cfg: ConfigDict,
        prompt_template: ConfigDict, # Expecting a dict with 'template_text' and 'input_columns'
        dataset_cfg: Optional[ConfigDict] = None, # Less critical now, primarily uses test_set
        pred_postprocessor: Optional[callable] = None, # Function to postprocess predictions
        dict_postprocessor: Optional[callable] = None, # Function to postprocess judge's raw output dict
        output_path: str = "./eval_results/generic_eval.json", # Default output path
        max_retries: int = 3,
        retry_delay: int = 10,
        keep_predictions: bool = False, # Added from original, though not heavily used here
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.judge_cfg = judge_cfg
        self.prompt_template_text = prompt_template.template # a string with placeholders like {query}, {prediction}
        self.prompt_input_columns = prompt_template.get('input_columns', ['query', 'prediction', 'reference'])


        self.dataset_cfg = dataset_cfg # Retained but its role is diminished
        self.pred_postprocessor = pred_postprocessor
        self.dict_postprocessor = dict_postprocessor
        self.output_path = output_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.keep_predictions = keep_predictions # Retained from original
        self.judge_system_prompt_content = judge_cfg.get('system_prompt_content', None) # Added for system/developer prompt


        self._out_dir = os.path.dirname(self.output_path)
        os.makedirs(self._out_dir, exist_ok=True)

        # Initialize OpenAI client
        # API key and base should be set as environment variables: OPENAI_API_KEY, OPENAI_API_BASE
        # Or passed directly in judge_cfg (though env vars are more common)
        self.openai_client = openai.OpenAI(
            api_key=judge_cfg.get('key', os.getenv("OPENAI_API_KEY")),
            base_url=judge_cfg.get('openai_api_base', os.getenv("OPENAI_API_BASE"))
        )
        self.judge_model_name = judge_cfg.get('path', judge_cfg.get('model', 'gpt-4o')) # 'path' is common in OC, 'model' is openai lib
        self.judge_temperature = judge_cfg.get('temperature', 0.0)
        self.judge_max_tokens = judge_cfg.get('max_out_len', 512)
        self.judge_query_per_second = judge_cfg.get('query_per_second', 1) # Simple rate limiting


    def _format_prompt(self, data_item: Dict[str, Any]) -> str:
        """Formats the prompt string using data_item."""
        # Ensure all necessary columns for the prompt are present in the data_item
        # Missing columns will be replaced with a placeholder string.
        prompt_values = {col: data_item.get(col, f"[{col} not found]") for col in self.prompt_input_columns}
        try:
            return self.prompt_template_text.format(**prompt_values)
        except KeyError as e:
            self.logger.error(f"Missing key {e} in prompt formatting for item: {data_item}. Input columns: {self.prompt_input_columns}")
            # Return a modified prompt indicating the error, or raise
            return f"Error: Missing key {e} for prompt. Data: { {k:v for k,v in prompt_values.items() if k in self.prompt_input_columns} }"


    def _get_judge_response(self, prompt: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                # Simple delay based on QPS, more sophisticated rate limiting might be needed for high volume
                time.sleep(1.0 / self.judge_query_per_second)

                messages_for_api = []
                if self.judge_system_prompt_content:
                    messages_for_api.append({"role": "system", "content": self.judge_system_prompt_content})
                messages_for_api.append({"role": "user", "content": prompt})

                response = self.openai_client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=messages_for_api,
                    temperature=self.judge_temperature,
                    max_tokens=self.judge_max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff might be better
                else:
                    self.logger.error("Max retries reached for OpenAI API call.")
                    return f"Error: Max retries reached for judge. Last error: {e}"
        return None # Should be unreachable if max_retries > 0

    def score(
        self,
        predictions: List[str], # Model's predictions
        references: Optional[List[str]] = None, # Ground truth references (can be part of test_set)
        test_set: Optional[Dataset] = None, # HF Dataset with columns for prompt
    ) -> Dict:
        """
        Score predictions using an LLM judge.

        Args:
            predictions: List of model predictions.
            references: List of ground truths. (Optional if all info is in test_set)
            test_set: Huggingface Dataset containing all necessary columns to format the judge's prompt.
                      Typically includes 'query', 'prediction', and 'reference'.
                      The 'prediction' column from this test_set will be IGNORED; the passed 'predictions' list is used.
        """
        if test_set is None:
            if references is None or len(predictions) != len(references):
                raise ValueError("If test_set is not provided, references must be provided and match length of predictions.")
            # Construct a minimal test_set if not provided
            # This assumes 'query' is not needed or is implicitly handled by the prompt template
            # and 'prediction' is from the arg, 'reference' from the arg.
            # A 'query' column is typically expected for most judge prompts.
            # This path is less robust. Prefer providing a comprehensive test_set.
            self.logger.warning("test_set not provided. Constructing a minimal dataset. Ensure your prompt_template does not require columns beyond 'prediction' and 'reference', or ensure 'query' etc., are part of the prompt_template_text itself if static.")
            data_for_dataset = {'prediction': predictions, 'reference': references}
            # Try to add a dummy query if 'query' is in prompt_input_columns
            if 'query' in self.prompt_input_columns:
                 data_for_dataset['query'] = ["N/A" for _ in predictions]
            test_set = Dataset.from_dict(data_for_dataset)

        if len(predictions) != len(test_set):
            raise ValueError(f"Length of predictions ({len(predictions)}) must match length of test_set ({len(test_set)}).")

        all_results = []
        processed_predictions = self.pred_postprocess(predictions)

        for i, example in enumerate(test_set):
            # Override the 'prediction' in the example with the one from the processed_predictions list
            current_data_item = example.copy() # Make a mutable copy
            current_data_item['prediction'] = processed_predictions[i]
            # Ensure 'reference' is also present if needed by prompt
            if 'reference' not in current_data_item and references and i < len(references):
                 current_data_item['reference'] = references[i]


            prompt_text = self._format_prompt(current_data_item)
            if prompt_text.startswith("Error:"): # Handle prompt formatting errors
                judge_response_text = prompt_text # Propagate error
            else:
                judge_response_text = self._get_judge_response(prompt_text)

            if judge_response_text is None: # Should not happen if _get_judge_response returns error string
                judge_response_text = "Error: Failed to get response from judge."

            # The dict_postprocessor is responsible for parsing the judge_response_text
            # (e.g., extracting a score, explanation) into a dictionary.
            # If None, the raw text is kept under a 'judge_response' key.
            if self.dict_postprocessor:
                # The postprocessor might expect the raw response and other context.
                # Adjust signature if it needs more than just the text.
                # For now, assume it takes the text and returns a dict.
                try:
                    eval_result = self.dict_postprocessor(judge_response_text)
                    if not isinstance(eval_result, dict): # Ensure it's a dict
                        eval_result = {'judge_output': eval_result, 'raw_judge_response': judge_response_text}
                except Exception as post_e:
                    self.logger.error(f"Error in dict_postprocessor for judge response '{judge_response_text[:100]}...': {post_e}")
                    eval_result = {'error_postprocessing': str(post_e), 'raw_judge_response': judge_response_text}

            else:
                eval_result = {'raw_judge_response': judge_response_text}

            # Include original data for reference, if desired (controlled by keep_predictions or similar logic)
            # For now, just keeping the judge's processed output and raw response.
            # The 'id' or index might be useful to include.
            # The OpenCompass format often has a 'details' structure.
            # We'll create a simpler list of results here.
            # The calling script (evaluation.py) will be responsible for aggregating.
            
            # Add original query, model prediction, reference for context in output
            result_item = {
                'id': example.get('id', i), # Use 'id' if present in dataset, else index
                'query': example.get('query', 'N/A'),
                'model_prediction': processed_predictions[i],
                'reference': example.get('reference', 'N/A' if not references else references[i]),
                'judge_evaluation': eval_result # This contains score etc. from postprocessor
            }
            all_results.append(result_item)

        # Save all raw judge outputs and processed scores
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4)
            self.logger.info(f"Saved detailed judge outputs to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save judge outputs to {self.output_path}: {e}")

        # The final score aggregation (e.g., averaging scores) should be done
        # by the caller (evaluation.py) based on the contents of `all_results`.
        # This `score` method returns the detailed results for flexibility.
        # For compatibility with how evaluation.py uses it, we can return a dict with 'details'
        # and a summary if the postprocessor provides a clear numeric score.
        
        summary_scores = []
        for res_item in all_results:
            # Assuming dict_postprocessor (like TextSummarizer) puts a 'score' key in judge_evaluation
            if 'judge_evaluation' in res_item and isinstance(res_item['judge_evaluation'], dict) and 'score' in res_item['judge_evaluation']:
                try:
                    summary_scores.append(float(res_item['judge_evaluation']['score']))
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not parse score from: {res_item['judge_evaluation']['score']}")

        final_output = {"details": all_results}
        if summary_scores:
            final_output["average_score"] = sum(summary_scores) / len(summary_scores) if summary_scores else 0
            final_output["num_scored"] = len(summary_scores)
        
        return final_output


    def pred_postprocess(self, predictions: List[str]) -> List[str]:
        if self.pred_postprocessor is None:
            return predictions
        else:
            # Assuming pred_postprocessor is a callable that takes a string and returns a string
            return [self.pred_postprocessor(pred) for pred in predictions]

    # Removed output_postprocess as its logic is now more integrated or handled by dict_postprocessor
    # and final aggregation in the calling script.

    # Removed default_judge_cfg property, as judge config is now directly passed and processed in __init__.
    # The calling script (evaluation.py) will be responsible for constructing this judge_cfg.