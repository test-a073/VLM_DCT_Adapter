from prettyprinter import pprint
from trl import SFTConfig
import torch 
from datasets import load_dataset
from peft import LoraConfig, get_peft_model 
import transformers
from torch.utils.data import DataLoader 

## DATASET 
# GQA Dataset - https://huggingface.co/datasets/lmms-lab/GQA
N_DATA_SAMPLES = 50
DATA_CACHE_DIR = "data/gqa_hf/"
DEBUG = True

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if N_DATA_SAMPLES is not None:
    val_balanced_instructions_ds = load_dataset("lmms-lab/GQA","val_balanced_instructions", cache_dir=DATA_CACHE_DIR,split="val").select(range(N_DATA_SAMPLES))
    train_balanced_instructions_ds = load_dataset("lmms-lab/GQA","train_balanced_instructions",cache_dir=DATA_CACHE_DIR,split="train").select(range(N_DATA_SAMPLES))
elif N_DATA_SAMPLES is None:
    val_balanced_instructions_ds = load_dataset("lmms-lab/GQA","val_balanced_instructions", cache_dir=DATA_CACHE_DIR,split="val")
    train_balanced_instructions_ds = load_dataset("lmms-lab/GQA","train_balanced_instructions",cache_dir=DATA_CACHE_DIR,split="train")

train_balanced_images_ds = load_dataset("lmms-lab/GQA","train_balanced_images", cache_dir=DATA_CACHE_DIR, split="train")
val_balanced_images_ds = load_dataset("lmms-lab/GQA","val_balanced_images", cache_dir=DATA_CACHE_DIR, split="val")

# Code from "https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl"
system_message = """You are a Vision Language Model specialized in interpreting visual data from Scene Graphs
Your task is to analyze the provided image and respond to queries with concise answers, usually a single word, number, or short phrase.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_gqa_data(sample, image_ds):
    # sample_id = sample['id']
    # sample_a_full = sample['fullAnswer'] 
    sample_img_id = sample['imageId']
    sample_img_id_index = image_ds['id'].index(sample_img_id)
    sample_img = val_balanced_images_ds[sample_img_id_index]['image']
    
    
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample_img, # PIL image
                },
                {
                    "type": "text",
                    "text": sample["question"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answer"]}],
        },
    ]

train_dataset = [format_gqa_data(sample, train_balanced_images_ds) for sample in train_balanced_instructions_ds]
val_dataset = [format_gqa_data(sample, val_balanced_images_ds) for sample in val_balanced_instructions_ds]

if DEBUG:
    print("Datasets Loaded")

## LOAD MODELS
# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_CACHE_DIR = "models/llama_3_vision_instruct/" 
HF_TOKEN = "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"
llama_3_vision_processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", token=HF_TOKEN,cache_dir=MODEL_CACHE_DIR )
llama_3_vision_model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct", token=HF_TOKEN, cache_dir=MODEL_CACHE_DIR)

## TEST MODEL BEFORE TRAINING
sample = val_dataset[0]
def generate_output_from_sample_llama_3_vision(sample, max_new_tokens=1024):
    
    # Add start end words for the text 
    text_input = llama_3_vision_processor.apply_chat_template(         

        sample[1:2], # sample - Use the sample without the system emssage
        tokenize=False, 
        add_generation_prompt=True, 
    ) 

    model_inputs = llama_3_vision_processor(
        text=text_input,
        images=sample[1]['content'][0]['image'],  # PIL.Image
        return_tensors="pt"
    ).to(llama_3_vision_model.device)

    with torch.inference_mode():
        generated_ids = llama_3_vision_model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        # Trim the generated ids to remove the input ids
        trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

        # Decode the output text
        output_text = llama_3_vision_processor.batch_decode(
            trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Question & Image :\n", sample[1:2])
        print("Model Output :\n", output_text[0])
        return output_text[0]  # Return the first decoded output text

generate_output_from_sample_llama_3_vision(sample)

## LORA CONFIGURATOIN
llama_3_lora_config = LoraConfig(
    r = 16, 
    lora_alpha=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
    lora_dropout=0.1,
    # bias=None
    # modules_to_save=[] # Modules apart from LoRA adapters to be saved,
)


# Create a data collator to encode text and image pairs
def collate_fn_llama_3_vision(samples):
    # print("samples")
    # pprint(samples)
    
    # Get the texts and images, and apply the chat template
    texts = [
        llama_3_vision_processor.apply_chat_template(sample[1:2], tokenize=False) for sample in samples
    ]  # Prepare texts for processing

    # print("TEXTS : ")
    # pprint(texts)
    image_inputs = [sample[1]['content'][0]['image'] for sample in samples]  # PIL images

    # print("IMAGES :")
    # pprint(image_inputs)

    # Tokenize the texts and process the images
    batch = llama_3_vision_processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == llama_3_vision_processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(llama_3_vision_processor, transformers.models.mllama.processing_mllama.MllamaProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [llama_3_vision_processor.tokenizer.convert_tokens_to_ids(llama_3_vision_processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch



BATCH_SIZE = 1
# NUM_WORKERS = 2 # 0
# HERE THE FOLLOWING LINES ARE COMMENTED BECAUSE I PASS DATASET DIRECTLY TO SFTTrainer
# train_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_llama_3_vision, num_workers=NUM_WORKERS, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_llama_3_vision, num_workers=NUM_WORKERS, shuffle=False) 

import wandb
WANDB_API = "0ee9657c6840ea93d9ff77d6eb2afcbc77b1a47c"

from huggingface_hub import login
HF_WRITE_TOKEN = "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"

login(HF_WRITE_TOKEN)


## TODO:TRAINING 
from trl import SFTTrainer

from trl import SFTConfig

# Configure training arguments
training_args = SFTConfig(
    output_dir="models/llama-3-vision-11b-instruct-trl-sft-GQA",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=BATCH_SIZE,  # Batch size for training
    per_device_eval_batch_size=BATCH_SIZE,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency

    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler

    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training

    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup

    # Hub and reporting
    push_to_hub=True,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics

    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing

    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
)
wandb.init(
    project="llama3-vision-11b-instruct-trl-sft-GQA",  # change this
    name="llama3-vision-11b-instruct-trl-sft-GQA",  # change this
    config=training_args,
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset

trainer = SFTTrainer(
    model=llama_3_vision_model,
    args=training_args,
    train_dataset=val_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn_llama_3_vision,
    peft_config=llama_3_lora_config,
    # tokenizer=llama_3_vision_processor.tokenizer,
)

trainer.train()

trainer.save_model(training_args.output_dir)