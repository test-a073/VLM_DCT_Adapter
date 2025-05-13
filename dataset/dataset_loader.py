from datasets import load_dataset
from torch.utils.data import DataLoader
import aiohttp
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoProcessor, AutoModelForCausalLM
import yaml
import torch
from torch.utils.data import Dataset

from qwen_vl_utils import process_vision_info

# To be put inside config
# model_id = "Qwen/Qwen2-VL-7B-Instruct"
# processor = Qwen2VLProcessor.from_pretrained(model_id)
model_id = None 
processor = None
config = None
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    # print("config",config)

    if config['models'][0]['name'] == 'qwen':
        # print("It is Qwen")
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        processor = Qwen2VLProcessor.from_pretrained(model_id)
    elif config['models'][0]['name'] == 'florence':
        processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base-ft", 
        trust_remote_code=True, 
        # revision=config.BASE_MODEL_REVISION
        )
    elif config['models'][0]['name'] == 'florence-large':
        processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large-ft", 
        trust_remote_code=True, 
        # revision=config.BASE_MODEL_REVISION
        )

def collate_fn_florence(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(
        text=list(questions), images=list(images), return_tensors="pt", padding=True
    )#.to(config.DEVICE, config.TORCH_DTYPE)
    return inputs, answers
def collate_fn_train(examples):
    # Get the texts and images, and apply the chat template
    
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # print("texts",texts)
    # Tokenize the texts and process the images
    # prompt_only_texts = []
    # output_texts = []
    # for text in texts:
    #     if "assistant\n" in text:
    #         prompt_only = text.split("assistant\n")[0] + "assistant\n"
    #     else:
    #         prompt_only = text  # fallback
    #     prompt_only_texts.append(prompt_only)
    #     output_texts.append(text.split("assistant\n")[1])
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    # prompt_only_batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True) 
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch
    # batch['ref'] = output_texts


    return batch  # Return the prepared batch

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # print("texts",texts)
    # Tokenize the texts and process the images
    prompt_only_texts = []
    output_texts = []
    for text in texts:
        if "assistant\n" in text:
            prompt_only = text.split("assistant\n")[0] + "assistant\n"
        else:
            prompt_only = text  # fallback
        prompt_only_texts.append(prompt_only)
        output_texts.append(text.split("assistant\n")[1])
    batch = processor(text=prompt_only_texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    # prompt_only_batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True) 
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch
    batch['ref'] = output_texts


    return batch  # Return the prepared batch

def format_chartqa_data(sample):
    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images. Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase. The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text. Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
    ]

def format_gqa_data(sample, image_ds):
    # sample_id = sample['id']
    # sample_a_full = sample['fullAnswer'] 
    sample_img_id = sample['imageId']
    sample_img_id_index = image_ds['id'].index(sample_img_id)
    sample_img = image_ds[sample_img_id_index]['image']
    
    
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


class ChartQAFlorenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<ChartQA>" + example['query']
        first_answer = example['label'][0]
        image = example['image'].convert("RGB")
        return question, first_answer, image

class GQADataset(Dataset):
    def __init__(self, data, images_data):
        self.data = data
        self.images_data = images_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<GQA>" + example["question"]
        
        first_answer = example["answer"][0]

        image_id = example['imageId']
        image_id_index = self.images_data['id'].index(image_id)
        image = self.images_data[image_id_index]['image']
        image = image.convert("RGB")
        return question, first_answer, image 

class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<DocVQA>" + example['question']
        first_answer = example['answers'][0]
        image = example['image'].convert("RGB")
        return question, first_answer, image

def load_dataset_vlm(dataset_name):
    if dataset_name == "docvqa":
        dataset_id = "HuggingFaceM4/DocumentVQA"
        data = load_dataset(dataset_id)

        if config['models'][0]['name'] == 'qwen':
            # TODO: SHOULD UPDATE THE FOLLOWING WITH A NEW COLLATE FUNCTION
            dataset = None
        
        elif config['models'][0]['name'] in  ('florence', 'florence-large'):
            BATCH_SIZE = config['train']['batch_size']
            NUM_WORKERS = config['train']['num_workers']
            print("Dataset ", config['models'][0]['name'], "loaded..")
            train_dataset = DocVQADataset(train_balanced_instructions_ds, train_balanced_images_ds)
            val_dataset = DocVQADataset(val_balanced_instructions_ds, val_balanced_images_ds)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate_fn_florence)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config['train']['batch_size'], shuffle=False, collate_fn=collate_fn_florence)
            dataset = {"train": train_loader,"val":val_loader,}

        return dataset, processor

    elif dataset_name == "gqa":
        
        dataset_id = "lmms-lab/GQA"
        N_DATA_SAMPLES = 50
        if N_DATA_SAMPLES is not None:
            val_balanced_instructions_ds = load_dataset(dataset_id,"val_balanced_instructions",split="val").select(range(N_DATA_SAMPLES))
            train_balanced_instructions_ds = load_dataset(dataset_id,"train_balanced_instructions",split="train").select(range(N_DATA_SAMPLES))
        elif N_DATA_SAMPLES is None:
            val_balanced_instructions_ds = load_dataset(dataset_id,"val_balanced_instructions",split="val")
            train_balanced_instructions_ds = load_dataset(dataset_id,"train_balanced_instructions",split="train")

        train_balanced_images_ds = load_dataset(dataset_id,"train_balanced_images", split="train")
        val_balanced_images_ds = load_dataset(dataset_id,"val_balanced_images", split="val")

        if config['models'][0]['name'] == 'qwen':
            train_dataset = [format_gqa_data(sample, train_balanced_images_ds) for sample in train_balanced_instructions_ds]
            val_dataset = [format_gqa_data(sample, val_balanced_images_ds) for sample in val_balanced_instructions_ds]
            dataset = {"train": train_dataset,"val":val_dataset}
        
        elif config['models'][0]['name'] in  ('florence', 'florence-large'):
            BATCH_SIZE = config['train']['batch_size']
            NUM_WORKERS = config['train']['num_workers']
            
            train_dataset = GQADataset(train_balanced_instructions_ds, train_balanced_images_ds)
            val_dataset = GQADataset(val_balanced_instructions_ds, val_balanced_images_ds)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate_fn_florence)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config['train']['batch_size'], shuffle=False, collate_fn=collate_fn_florence)
            dataset = {"train": train_loader,"val":val_loader}

        return dataset, processor

    elif dataset_name=="chart_qa":
        dataset_id = "HuggingFaceM4/ChartQA"
        train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])
        if config['models'][0]['name'] == 'qwen':
            train_dataset = [format_chartqa_data(sample) for sample in train_dataset]
            eval_dataset = [format_chartqa_data(sample) for sample in eval_dataset]
            test_dataset = [format_chartqa_data(sample) for sample in test_dataset]
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,collate_fn=collate_fn_train)
            val_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True,collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,collate_fn=collate_fn)
            dataset = {"train": train_loader,"val":val_loader, "test":test_loader}
        elif config['models'][0]['name'] == 'florence':
            train_dataset = ChartQAFlorenceDataset(train_dataset)
            eval_dataset = ChartQAFlorenceDataset(eval_dataset)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate_fn_florence)
            val_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=config['train']['batch_size'], shuffle=False, collate_fn=collate_fn_florence)
            dataset = {"train": train_loader,"val":val_loader,}
        elif config['models'][0]['name'] == 'florence-large':
            train_dataset = ChartQAFlorenceDataset(train_dataset)
            eval_dataset = ChartQAFlorenceDataset(eval_dataset)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config['train']['batch_size'], shuffle=True, collate_fn=collate_fn_florence)
            val_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=config['train']['batch_size'], shuffle=False, collate_fn=collate_fn_florence)
            dataset = {"train": train_loader,"val":val_loader,}
        return dataset,processor
    elif dataset_name == "vqa_v2":
        # dataset = load_dataset("visual_genome", "question_answers_v1.2.0",storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
        dataset = load_dataset("Graphcore/vqa", split="training[:50]")
        # print(dataset['train'][0])
        # dataset = dataset['train']
        # dataset = dataset["train"].map(lambda e: {"inputs": e["question"], "labels": e["answer"]})
    elif dataset_name == "ok_vqa":
        dataset = load_dataset("ok_vqa")
        dataset = dataset["train"].map(lambda e: {"inputs": e["question"], "labels": e["answer"]})
    elif dataset_name == "coco_captions":
        dataset = load_dataset("coco_captions")
        dataset = dataset["train"].map(lambda e: {"inputs": e["image_id"], "labels": e["caption"]})
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return dataset