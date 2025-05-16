import os
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset

# Assuming the refactored GenericLLMEvaluator is in the same directory
from generic_evaluator import GenericLLMEvaluator

# Removed OpenCompass specific imports:
# import mmengine
# from opencompass.openicl.icl_evaluator import BaseEvaluator
# from opencompass.registry import ICL_EVALUATORS
# from opencompass.utils.logging import get_logger

# If mmengine.ConfigDict is used for config, ensure it's available or use plain dicts.
# For consistency with GenericLLMEvaluator, let's add the ConfigDict fallback.
try:
    from mmengine.config import ConfigDict
except ImportError:
    logging.warning("mmengine.config.ConfigDict not found for CascadeEvaluator. Using a basic dict wrapper.")
    class ConfigDict(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        def __setattr__(self, name, value):
            self[name] = value

class CascadeEvaluator: # Removed inheritance from BaseEvaluator
    """Cascade Evaluator - Refactored to be OpenCompass independent.

    First uses a rule-based method or a sample scoring function to judge predictions.
    If a sample is marked as incorrect or if in parallel mode,
    then it uses an LLM judge (refactored GenericLLMEvaluator) to re-evaluate it.
    """

    def __init__(
        self,
        llm_evaluator_cfg: ConfigDict, # Config for the refactored GenericLLMEvaluator
        rule_evaluator_cfg: Optional[ConfigDict] = None, # Config for a rule-based evaluator (needs to be standalone too)
        sample_score_fn: Optional[Callable[[str, str], Dict]] = None, # Function: (pred, ref) -> {score_key: val, ...}
        parallel: bool = True, # If true, LLM judge runs on all samples, not just rule-failed ones
        output_path: str = "./eval_results/cascade_eval.json",
        rule_evaluator_class: Optional[type] = None, # Allow passing the class for rule_evaluator
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._out_dir = os.path.dirname(output_path)
        self.output_path = output_path
        os.makedirs(self._out_dir, exist_ok=True)

        # Initialize the LLM evaluator (refactored GenericLLMEvaluator)
        # Ensure llm_evaluator_cfg contains all necessary args for GenericLLMEvaluator's __init__
        # (judge_cfg, prompt_template, output_path etc.)
        # We might need to adjust how output_path is handled for the inner LLM evaluator.
        llm_output_subpath = os.path.join(self._out_dir, os.path.basename(self.output_path).replace(".json", "_llm_judge.json"))
        if 'output_path' not in llm_evaluator_cfg:
             llm_evaluator_cfg['output_path'] = llm_output_subpath

        self.llm_evaluator = GenericLLMEvaluator(**llm_evaluator_cfg)

        # Initialize the rule evaluator if provided
        self.rule_evaluator = None
        if rule_evaluator_cfg:
            if rule_evaluator_class is None:
                # This part is tricky without a registry or a known standalone rule evaluator structure.
                # For now, we'll raise an error if cfg is given but class is not.
                # User would need to pass a class that can be instantiated with rule_evaluator_cfg.
                raise ValueError("rule_evaluator_cfg provided but rule_evaluator_class is missing.")
            try:
                self.rule_evaluator = rule_evaluator_class(**rule_evaluator_cfg)
            except Exception as e:
                self.logger.error(f"Failed to initialize rule_evaluator_class: {e}")
                raise
        
        self.sample_score_fn = sample_score_fn
        self.parallel = parallel

        if not self.rule_evaluator and not self.sample_score_fn:
            self.logger.warning(
                "CascadeEvaluator initialized without rule_evaluator or sample_score_fn. "
                "It will behave like the configured LLM evaluator directly on all samples if parallel=True, "
                "or skip rule-based phase if parallel=False (effectively same as LLM evaluator then)."
            )
            # If parallel is False and no rule/sample_fn, it means LLM will only run if something "failed" which won't happen.
            # So, if no rule-based part, parallel should ideally be True to pass all to LLM.
            if not self.parallel:
                self.logger.info("No rule/sample_fn and parallel=False. Forcing parallel=True to ensure LLM evaluation runs.")
                self.parallel = True 


    def _score_with_rule_or_sample_fn(self, prediction: str, reference: str) -> Dict[str, Any]:
        """Score a single sample using sample_score_fn or rule_evaluator."""
        if self.sample_score_fn:
            result = self.sample_score_fn(prediction, reference)
            if not isinstance(result, dict):
                result = {'correct': bool(result), 'score_source': 'sample_fn'} # Basic wrapping
            else:
                result['score_source'] = 'sample_fn'
            return result
        elif self.rule_evaluator:
            # Assuming rule_evaluator has a .score() method that takes lists
            # and returns a dict with a 'details' list or an 'accuracy'-like key.
            # This is a placeholder for how a standalone rule_evaluator might work.
            # The exact API of a custom rule_evaluator would need to be defined.
            try:
                # This part is highly dependent on the rule_evaluator's API.
                # Let's assume it can score a single sample or we adapt.
                # For simplicity, if it has a method like `evaluate_single`:
                if hasattr(self.rule_evaluator, 'evaluate_single'):
                     result = self.rule_evaluator.evaluate_single(prediction, reference)
                # Or if it expects lists like OpenCompass evaluators:
                elif hasattr(self.rule_evaluator, 'score'):
                     # This is less ideal for single samples due to overhead
                     rule_eval_output = self.rule_evaluator.score([prediction], test_set=Dataset.from_dict({'prediction': [prediction], 'reference': [reference]}))
                     # Extract single result - this depends on rule_eval_output structure
                     if 'details' in rule_eval_output and len(rule_eval_output['details']) > 0:
                         result = rule_eval_output['details'][0]
                     elif 'accuracy' in rule_eval_output : # e.g. simple accuracy
                         result = {'correct': rule_eval_output['accuracy'] > 0} 
                     else: # Fallback
                         result = {'correct': False, 'error': 'Unknown rule_evaluator output format'}
                else:
                    raise NotImplementedError("Rule evaluator does not have a recognized scoring method.")
                
                if not isinstance(result, dict):
                     result = {'correct': bool(result)} # Basic wrapping
                result['score_source'] = 'rule_evaluator'
                return result

            except Exception as e:
                self.logger.error(f"Error in rule_evaluator: {e}")
                return {'correct': False, 'error': str(e), 'score_source': 'rule_evaluator'}
        else:
            # This case should be handled by the __init__ logic making parallel=True
            # If we reach here, it means no rule-based scoring is possible.
            # We return a neutral result indicating it should go to LLM if parallel=True.
            return {'correct': False, 'score_source': 'none', 'note': 'No rule/sample_fn configured'}

    def _get_llm_correctness(self, llm_detail: Dict[str, Any]) -> bool:
        """Determine if the LLM judge considers the answer correct based on its output."""
        # This function needs to be adapted based on how the refactored GenericLLMEvaluator
        # and its dict_postprocessor structure the 'judge_evaluation' output.
        # Assuming the dict_postprocessor for GenericLLMEvaluator might produce a 'score' or 'correct' field.
        judge_eval = llm_detail.get('judge_evaluation', {})
        if 'correct' in judge_eval:
            return bool(judge_eval['correct'])
        if 'score' in judge_eval: # Assuming higher score means correct
            try:
                # Handle scores like "7/10" or just numbers
                score_val_str = str(judge_eval['score']).split('/')[0].strip()
                score_val = float(score_val_str)
                # Define a threshold for correctness, e.g., > 5 for a 1-10 scale
                # This threshold is arbitrary and should be configured or made more robust.
                return score_val > 5 # Example threshold for a 1-10 scale
            except ValueError:
                # Handle cases where score is not easily convertible to float (e.g., text like "Good")
                # A more robust way is to have the judge output a clear boolean or categorical rating.
                pass 
        # Fallback: if the raw judge response contains certain keywords (very brittle)
        raw_response = str(judge_eval.get('raw_judge_response', '')).strip().upper()
        if raw_response.startswith('A') or "CORRECT" in raw_response or "YES" in raw_response : # from old logic
             return True
        return False # Default to False if correctness cannot be determined

    def score(
        self,
        predictions: List[str],
        references: List[str], # Should align with predictions
        test_set: Optional[Dataset] = None, # HF Dataset with columns for prompt, must align with predictions/references
    ) -> Dict[str, Any]:
        self.logger.info(f"Running CascadeEvaluator in {'parallel' if self.parallel else 'cascade'} mode.")

        if len(predictions) != len(references):
            raise ValueError("Length of predictions and references must match.")
        if test_set and len(predictions) != len(test_set):
            raise ValueError("Length of predictions and test_set must match if test_set is provided.")

        # Create a base test_set if not provided, for llm_evaluator compatibility
        if not test_set:
            # We need 'query' for the default llm_evaluator prompt typically.
            # This is a minimal construction.
            self.logger.warning("CascadeEvaluator: test_set not provided. Creating minimal dataset for LLM evaluator. Queries will be 'N/A'.")
            ds_dict = {
                'prediction': predictions, # This will be overridden by LLM evaluator with its input predictions
                'reference': references,
                'query': ['N/A'] * len(predictions) # Placeholder query
            }
            test_set = Dataset.from_dict(ds_dict)

        detailed_results = []
        llm_eval_predictions = []
        llm_eval_references = [] # References for the LLM evaluator
        llm_eval_indices_in_test_set = [] # original indices from the main test_set for failed items

        # Phase 1: Rule-based or sample_score_fn evaluation
        initial_correct_count = 0
        if self.rule_evaluator or self.sample_score_fn:
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                rule_result = self._score_with_rule_or_sample_fn(pred, ref)
                is_correct_by_rule = rule_result.get('correct', False)
                if is_correct_by_rule and not self.parallel : # If correct by rule and not in parallel mode, count and skip LLM
                    initial_correct_count += 1
                else: # Either failed rule, or in parallel mode (so LLM runs anyway)
                    llm_eval_predictions.append(pred)
                    llm_eval_references.append(ref)
                    llm_eval_indices_in_test_set.append(i)
                
                # Store rule result associated with original index
                # The structure of detailed_results needs to accommodate both rule and potential LLM eval.
                # We initialize with rule eval, and will update with LLM eval later.
                current_detail = {
                    'id': test_set[i].get('id', i) if test_set else i,
                    'query': test_set[i].get('query','N/A') if test_set else 'N/A',
                    'prediction': pred,
                    'reference': ref,
                    'rule_evaluation': rule_result,
                    'final_correct': is_correct_by_rule if not self.parallel else None # Tentative, may be updated by LLM
                }
                detailed_results.append(current_detail)
            
            rule_accuracy = (100 * initial_correct_count / len(predictions)) if predictions else 0
            self.logger.info(f'Rule-based/Sample_fn evaluation: {initial_correct_count}/{len(predictions)} correct ({rule_accuracy:.2f}%). Will pass {len(llm_eval_predictions)} to LLM.')
        else: # No rule evaluator or sample_score_fn, all go to LLM if parallel=True (which __init__ enforces)
            self.logger.info("No rule-based/sample_fn configured. All samples will be evaluated by LLM.")
            llm_eval_predictions = predictions[:]
            llm_eval_references = references[:]
            llm_eval_indices_in_test_set = list(range(len(predictions)))
            # Initialize detailed_results for all items, to be populated by LLM eval
            for i, (pred,ref) in enumerate(zip(predictions,references)):
                detailed_results.append({
                    'id': test_set[i].get('id', i) if test_set else i,
                    'query': test_set[i].get('query','N/A') if test_set else 'N/A',
                    'prediction': pred,
                    'reference': ref,
                    'rule_evaluation': {'score_source': 'none'},
                    'final_correct': None
                })


        # Phase 2: LLM Evaluation for samples marked for it
        if llm_eval_predictions:
            # Construct the subset of the test_set for the LLM evaluator
            # This needs to have the columns expected by GenericLLMEvaluator's prompt template
            # (e.g., query, reference - prediction will come from llm_eval_predictions list)
            llm_test_set_subset = test_set.select(llm_eval_indices_in_test_set)
            
            self.logger.info(f'Running LLM evaluation on {len(llm_eval_predictions)} samples...')
            llm_eval_output = self.llm_evaluator.score(
                predictions=llm_eval_predictions, 
                # references for llm_evaluator are derived from the llm_test_set_subset or passed explicitly
                # if llm_evaluator's prompt needs 'reference', it should be in llm_test_set_subset
                test_set=llm_test_set_subset 
            )
            
            # Merge LLM results back into detailed_results
            # llm_eval_output['details'] is a list of dicts from GenericLLMEvaluator
            llm_output_details = llm_eval_output.get('details', [])
            if len(llm_output_details) != len(llm_eval_indices_in_test_set):
                self.logger.error("Mismatch between LLM output details and number of samples sent to LLM. Check LLM evaluator.")
            else:
                for i, llm_detail_item in enumerate(llm_output_details):
                    original_dataset_index = llm_eval_indices_in_test_set[i]
                    detailed_results[original_dataset_index]['llm_evaluation'] = llm_detail_item.get('judge_evaluation', {})
                    detailed_results[original_dataset_index]['llm_raw_output'] = llm_detail_item # Store full LLM item for debug
                    # Determine final correctness based on LLM evaluation if it happened for this item
                    is_correct_by_llm = self._get_llm_correctness(llm_detail_item) # llm_detail_item is the full dict from GenericLLM
                    detailed_results[original_dataset_index]['final_correct'] = is_correct_by_llm
        
        # Calculate final accuracy based on 'final_correct' field
        final_correct_count = sum(1 for dr in detailed_results if dr.get('final_correct', False))
        final_accuracy = (100 * final_correct_count / len(predictions)) if predictions else 0
        self.logger.info(f"CascadeEvaluator Final: {final_correct_count}/{len(predictions)} correct ({final_accuracy:.2f}%).")

        # Save detailed results
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=4)
            self.logger.info(f"Saved CascadeEvaluator detailed outputs to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save CascadeEvaluator outputs: {e}")

        return {
            "accuracy": final_accuracy,
            "total_correct": final_correct_count,
            "total_samples": len(predictions),
            "details": detailed_results
        }