import argparse
import json
import logging
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from tqdm import tqdm
import math # For DCTAdapter

# --- DCTAdapter Definition (copied from adapter/mistral_adapter.py) ---
class DCTAdapter(nn.Module):
    def __init__(self, input_dim=4096, num_components=24, tau=1.0): # num_components not used in current impl
        super().__init__()
        self.tau = tau
        # The intermediate dimension (18) is hardcoded here as per the provided adapter code
        # If num_components was meant to define this, the adapter code needs adjustment.
        self.adapter_intermediate_dim = 18 
        self.adapter_down = nn.Linear(input_dim, self.adapter_intermediate_dim, bias=False)
        self.adapter_up = nn.Linear(self.adapter_intermediate_dim, input_dim, bias=False)

    def create_dct_matrix(self, N, device=None, dtype=torch.float32):
        n_range = torch.arange(N, dtype=dtype, device=device).unsqueeze(0)
        k_range = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
        dct_matrix = torch.cos(math.pi * (2 * n_range + 1) * k_range / (2 * N))
        dct_matrix[0] *= 1 / math.sqrt(2)
        dct_matrix *= math.sqrt(2 / N)
        return dct_matrix

    def dct1(self, x):
        N = x.size(-1)
        dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
        return x @ dct_mat.T # Transpose for DCT-II

    def idct1(self, x):
        N = x.size(-1)
        # For IDCT-II, we use the DCT-III matrix, which is the transpose of DCT-II matrix (scaled appropriately)
        # However, standard IDCT is often implemented by re-using DCT matrix properties.
        # Given the original code, it seems to use the same matrix for idct, which implies DCT-II followed by DCT-III if not scaled
        # Or, if dct_mat is orthonormal (which it is, scaled), x @ dct_mat performs IDCT if x @ dct_mat.T was DCT
        dct_mat = self.create_dct_matrix(N, x.device, x.dtype)
        return x @ dct_mat # Using dct_mat directly as per original for inverse op

    def forward(self, hidden_states):
        # print(f"DCTAdapter: Input hidden_states shape: {hidden_states.shape}")
        original_dtype = hidden_states.dtype
        # Ensure calculations are in float32 if model is in float16/bfloat16 for stability
        # hidden_states_float32 = hidden_states.to(torch.float32)
        
        dct_transformed = self.dct1(hidden_states)
        
        # Reshape for linear layers: (batch_size * sequence_length, features)
        # b, s, f = dct_transformed.shape
        # z_reshaped = dct_transformed.reshape(-1, f)
        
        # Apply adapter layers
        # down_projected = self.adapter_down(z_reshaped)
        # activated = F.leaky_relu(down_projected)
        # up_projected = self.adapter_up(activated)
        
        # Reshape back to original: (batch_size, sequence_length, features)
        # adapter_output_reshaped = up_projected.view_as(dct_transformed)

        # Simpler path if adapter_down/up expect [B, S, F] directly (needs check on Linear layer behavior with 3D inputs)
        # Assuming Linear layers in PyTorch handle [..., in_features] -> [..., out_features]
        down_projected = self.adapter_down(dct_transformed)
        activated = F.leaky_relu(down_projected)
        up_projected = self.adapter_up(activated)

        idct_transformed = self.idct1(up_projected)
        
        return (hidden_states + idct_transformed).to(original_dtype)

# --- Helper Functions for Adapter Injection (copied) ---
def get_parent_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    names = name.split('.')
    parent = model
    for n in names[:-1]:
        parent_candidate = getattr(parent, n, None)
        if parent_candidate is None:
            raise AttributeError(f"Module {parent} does not have attribute {n} (from name {name})")
        parent = parent_candidate
    return parent

def inject_adapters(
    model: torch.nn.Module,
    adapter_cls: type,
    base_adapter_args: dict,
    layers_config: list
) -> torch.nn.Module:
    logger.info(f"Starting adapter injection with {adapter_cls.__name__}...")
    for layer_conf in layers_config:
        target_layer_name = layer_conf['name']
        try:
            parent_module = get_parent_module(model, target_layer_name)
            original_module = getattr(parent_module, target_layer_name.split('.')[-1])

            current_adapter_args = base_adapter_args.copy()
            
            # Ensure in_features for adapter matches the original module's output features
            if hasattr(original_module, 'out_features') and isinstance(original_module.out_features, int):
                actual_in_features = original_module.out_features
                # Override input_dim from base_adapter_args if layer_conf doesn't specify one
                current_adapter_args['input_dim'] = actual_in_features 
                logger.info(f"Dynamically setting adapter 'input_dim' for {target_layer_name} to {actual_in_features}")
            elif 'input_dim' not in current_adapter_args:
                logger.error(f"Cannot determine input_dim for adapter at {target_layer_name}. "
                               f"Original module {type(original_module)} lacks 'out_features'. "
                               f"Please specify 'input_dim' in adapter_params in config.")
                raise ValueError(f"Missing input_dim for {target_layer_name}")


            adapter_instance = adapter_cls(**current_adapter_args)
            # Replace the original module with a sequence: original -> adapter
            setattr(parent_module, target_layer_name.split('.')[-1], torch.nn.Sequential(original_module, adapter_instance))
            logger.info(f"Successfully injected adapter after {target_layer_name} with args: {current_adapter_args}")

        except AttributeError as e:
            logger.error(f"AttributeError while trying to inject adapter into {target_layer_name}: {e}. Skipping this layer.", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to inject adapter into {target_layer_name}: {e}", exc_info=True)
    return model

def freeze_model_except_adapters(model: torch.nn.Module):
    logger.info("Freezing model parameters except for adapter layers...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    for name, param in model.named_parameters():
        if 'adapter' in name.lower() or 'dctadapter' in name.lower(): # Check for adapter in name
            param.requires_grad = True
            trainable_params += param.numel()
            # logger.info(f"Unfrozen: {name} ({param.numel()})")
        else:
            param.requires_grad = False
    
    logger.info(f"Trainable adapter params: {trainable_params} || All params: {total_params} || Trainable %: {100 * trainable_params / total_params:.4f}")
    if trainable_params == 0:
        logger.warning("No parameters were marked as trainable. Check adapter naming and injection logic.")


# --- Training Data Preprocessing ---
def preprocess_gqa_for_training(examples, tokenizer, question_field, answer_field):
    prompts = []
    for q, a in zip(examples[question_field], examples[answer_field]):
        if q and a: # Ensure question and answer are not None or empty
            # Using a simple template, adjust if Mistral needs a specific chat/instruct template
            # Format: <s>[INST] Question [/INST] Answer </s>
            # For Causal LM, the entire sequence is usually fed as input_ids, and labels are input_ids shifted.
            # Here, we form one continuous text.
            text = f"<s>[INST] {q} [/INST] {a}</s>"
            prompts.append(text)
        else:
            prompts.append("") # Append empty string if data is missing, tokenizer will handle it or produce empty output

    # Tokenize the full texts. Labels will be the input_ids themselves for Causal LM.
    # Setting padding to 'max_length' and truncation=True.
    # max_length should be chosen carefully based on model capacity and VRAM.
    tokenized_outputs = tokenizer(
        prompts,
        truncation=True,
        padding="max_length", # Pad to max_length
        max_length=512,      # Example max_length
        return_attention_mask=True
    )
    # For Causal LM, labels are usually the input_ids.
    tokenized_outputs["labels"] = [ids.copy() for ids in tokenized_outputs["input_ids"]]
    return tokenized_outputs


# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCT Adapters on Mistral and generate outputs for AlpacaEval.")
    parser.add_argument("--base_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter_config_path", type=str, default="/common/home/users/s/sasikayw/scratchDirectory/VLM_DCT_Adapter/config/config.yaml")
    parser.add_argument("--training_dataset_name", type=str, default="gqa") # Assuming 'gqa', 'all' or similar HF identifier
    parser.add_argument("--training_dataset_config", type=str, default="all", help="Specify config for GQA if needed, e.g., 'real_train'")
    parser.add_argument("--gqa_question_field", type=str, default="question")
    parser.add_argument("--gqa_answer_field", type=str, default="answer")
    parser.add_argument("--adapted_model_name", type=str, default="mistral-instruct-adapted-dct")
    parser.add_argument("--alpaca_output_json_file", type=str, default=None, help="Defaults to <adapted_model_name>_outputs.json")
    parser.add_argument("--training_epochs", type=int, default=3) # Default from your config, adjust as needed
    parser.add_argument("--batch_size", type=int, default=1)    # Default from your config
    parser.add_argument("--learning_rate", type=float, default=1e-4) # Default from your config
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max samples for GQA training (debugging)")
    parser.add_argument("--max_alpaca_samples", type=int, default=None, help="Max samples for AlpacaEval generation (debugging)")
    parser.add_argument("--output_dir_training", type=str, default="./dct_adapter_mistral_training_gqa")
    parser.add_argument("--disable_training", action="store_true", help="Skip training and load model directly (expects already trained adapters)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, or mps. Auto-detects if None.")


    args = parser.parse_args()

    if args.alpaca_output_json_file is None:
        args.alpaca_output_json_file = f"{args.adapted_model_name}_outputs.json"
    
    if args.device:
        DEVICE = args.device
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Using device: {DEVICE}")

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer for {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")

    # 2. Load Base Model
    logger.info(f"Loading base model {args.base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, 
        # torch_dtype=torch.bfloat16 # Uncomment if VRAM is an issue and GPU supports it
    )

    # 3. Load Adapter Configuration from YAML
    logger.info(f"Loading adapter configuration from {args.adapter_config_path}")
    with open(args.adapter_config_path, 'r') as f:
        adapter_config_yaml = yaml.safe_load(f)
    
    adapter_params_from_yaml = adapter_config_yaml.get('adapter', {}).get('params', {})
    # Ensure 'input_dim' from yaml is correctly named 'input_dim' for DCTAdapter constructor
    # The DCTAdapter class expects 'input_dim'. The yaml has 'input_dim'.
    # num_components is also in yaml and DCTAdapter constructor.
    
    adapter_layers_from_yaml = adapter_config_yaml.get('adapter', {}).get('layers', [])
    
    if not adapter_layers_from_yaml:
        logger.error("No adapter layers specified in the YAML configuration. Exiting.")
        exit(1)
    if not adapter_params_from_yaml:
        logger.warning("No adapter parameters (params) specified in the YAML. Using defaults for DCTAdapter if any.")


    # 4. Inject Adapters
    logger.info("Injecting DCT Adapters...")
    model = inject_adapters(model, DCTAdapter, 
                            base_adapter_args=adapter_params_from_yaml, 
                            layers_config=adapter_layers_from_yaml)
    model.to(DEVICE)

    # 5. Training Phase
    if not args.disable_training:
        logger.info("Starting Training Phase for Adapters...")
        freeze_model_except_adapters(model) # Freeze base model, unfreeze adapters

        # Load GQA dataset
        logger.info(f"Loading GQA dataset ('{args.training_dataset_name}', config: '{args.training_dataset_config}')")
        try:
            # GQA might require specific loading for question/answer pairs if it's image-based
            # This assumes a text-only version or relevant text fields are available.
            # For official GQA, you might need to download files and process them.
            # Using a placeholder here - replace with actual GQA loading if different.
            # Common GQA on HF might be image-based. If you mean a purely text version, ensure it's findable.
            # For example, if it's a custom JSON/CSV:
            # train_dataset = load_dataset('json', data_files={'train': 'path/to/your/gqa_train.json'})['train']
            gqa_dataset = load_dataset(args.training_dataset_name, args.training_dataset_config, split="train") # Adjust split
        except Exception as e:
            logger.error(f"Failed to load GQA dataset: {e}. Ensure it's a text-based Q&A dataset accessible by Hugging Face 'datasets'.")
            logger.error("You might need to download GQA data manually and prepare it as a JSON/CSV file, then load with load_dataset('json', data_files=...). ")
            exit(1)

        if args.max_train_samples:
            logger.info(f"Selecting {args.max_train_samples} for GQA training.")
            gqa_dataset = gqa_dataset.select(range(min(args.max_train_samples, len(gqa_dataset))))

        logger.info("Preprocessing GQA dataset...")
        tokenized_gqa_dataset = gqa_dataset.map(
            lambda examples: preprocess_gqa_for_training(
                examples, tokenizer, args.gqa_question_field, args.gqa_answer_field
            ),
            batched=True,
            remove_columns=gqa_dataset.column_names
        )
        
        training_hparams = adapter_config_yaml.get('train', {})
        final_epochs = args.training_epochs if args.training_epochs is not None else training_hparams.get('epochs', 3)
        final_batch_size = args.batch_size if args.batch_size is not None else training_hparams.get('batch_size', 1)
        final_lr = args.learning_rate if args.learning_rate is not None else training_hparams.get('lr', 1e-4)


        logger.info(f"Training adapters with: epochs={final_epochs}, batch_size={final_batch_size}, lr={final_lr}")
        
        # Setup Trainer
        training_args_hf = TrainingArguments(
            output_dir=args.output_dir_training,
            num_train_epochs=final_epochs,
            per_device_train_batch_size=final_batch_size,
            learning_rate=final_lr,
            logging_dir=f"{args.output_dir_training}/logs",
            logging_steps=10, # Log more frequently for small datasets/batches
            save_strategy="epoch", # Save per epoch or steps
            # report_to="wandb", # If you use wandb
            fp16=torch.cuda.is_available(), # Use fp16 if on CUDA for speed/memory
            # Add other arguments as needed: warmup_steps, weight_decay, etc.
        )

        trainer = Trainer(
            model=model,
            args=training_args_hf,
            train_dataset=tokenized_gqa_dataset,
            tokenizer=tokenizer, # Important for padding collation
            # data_collator can be used for custom padding if needed
        )
        
        logger.info("Starting adapter training...")
        trainer.train()
        logger.info("Adapter training finished.")
        # To save only adapter weights (if not using PEFT which does this automatically):
        # adapter_state_dict = {k: v for k, v in model.state_dict().items() if 'adapter' in k.lower()}
        # torch.save(adapter_state_dict, os.path.join(args.output_dir_training, "final_dct_adapter_weights.pth"))
        # logger.info(f"Raw adapter weights saved to {os.path.join(args.output_dir_training, 'final_dct_adapter_weights.pth')}")
        # The trainer already saves the full model checkpoint, which includes the trained adapters.
        # To load it later: model = AutoModelForCausalLM.from_pretrained(args.output_dir_training/checkpoint-XYZ)
        # then inject adapters again if it's a raw base model checkpoint, or ensure the checkpoint is the adapted one.
        # For simplicity, we assume the `model` object in memory is now the one with trained adapters.

    else:
        logger.info("Skipping training phase as per --disable_training flag.")
        logger.info(f"Ensure that the model loaded ({args.base_model_id}) has pre-trained adapters, or this script was run previously with training.")
        # If you had saved adapter-only weights, you'd load them here:
        # adapter_weights_path = os.path.join(args.output_dir_training, "final_dct_adapter_weights.pth") 
        # if os.path.exists(adapter_weights_path):
        #    logger.info(f"Loading pre-trained adapter weights from {adapter_weights_path}")
        #    model.load_state_dict(torch.load(adapter_weights_path), strict=False) # strict=False as we only load adapter parts
        # else:
        #    logger.warning(f"--disable_training was set, but no adapter weights found at {adapter_weights_path}")


    # 6. Prepare for AlpacaEval Generation
    model.eval()
    logger.info("Model set to evaluation mode.")

    # 7. Load AlpacaEval Dataset
    logger.info("Loading AlpacaEval dataset.")
    alpaca_eval_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    if args.max_alpaca_samples:
        logger.info(f"Selecting {args.max_alpaca_samples} for AlpacaEval generation.")
        alpaca_eval_dataset = alpaca_eval_dataset.select(range(min(args.max_alpaca_samples, len(alpaca_eval_dataset))))

    # 8. Generate Outputs for AlpacaEval
    logger.info(f"Generating outputs for AlpacaEval dataset (for model: {args.adapted_model_name})...")
    alpaca_outputs_data = []
    for example in tqdm(alpaca_eval_dataset, desc="AlpacaEval Generation"):
        instruction = example["instruction"]
        
        # Using apply_chat_template for robust prompt formatting for instruct models
        # This assumes Mistral-Instruct's template is correctly handled by the tokenizer
        try:
            chat_template_input = [{"role": "user", "content": instruction}]
            prompt = tokenizer.apply_chat_template(chat_template_input, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw instruction as prompt.")
            prompt = instruction

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE) # Max input length for prompt
        
        with torch.no_grad():
            # max_new_tokens determines how long the generated response can be
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # For more deterministic outputs for evaluation
            )
        
        # Decoding: remove the prompt part from the generated output
        # generated_ids[0] contains the full sequence (prompt + generation)
        # inputs.input_ids.shape[1] gives the length of the prompt
        # Slicing generated_ids[0][inputs.input_ids.shape[1]:] gets only the generated tokens
        generated_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        alpaca_outputs_data.append({
            "instruction": instruction,
            "output": generated_text.strip(),
            "generator": args.adapted_model_name,
            "dataset": example.get("dataset", "alpaca_eval") # Include original dataset if present
        })

    # 9. Save AlpacaEval Outputs to JSON
    logger.info(f"Saving AlpacaEval outputs to {args.alpaca_output_json_file}")
    with open(args.alpaca_output_json_file, "w") as f:
        json.dump(alpaca_outputs_data, f, indent=2)
    
    logger.info("Script finished.")
    logger.info("Next step: Run AlpacaEval CLI to evaluate the generated outputs.")
    logger.info(f"Example command: alpaca_eval evaluate --model_outputs {args.alpaca_output_json_file} --annotators_config weighted_alpaca_eval_gpt4_turbo --name {args.adapted_model_name}") 