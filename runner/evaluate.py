from torch.utils.data import DataLoader
from tqdm import tqdm
from runner.metrics import average_normalized_levenshtein_similarity
import torch

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

# def evaluate_model(model, dataset, processor, preprocess_fn=None, batch_size=8, device="cuda"):
#     loader = dataset['val']
#     model.eval()
#     model.to(device)

#     total_correct = 0
#     total_samples = 0

#     for batch in tqdm(loader, desc="Evaluating"):
#         inputs = preprocess_fn(batch) if preprocess_fn else batch
#         labels = batch["labels"].to(device)
#         inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
#         with torch.no_grad():
#             outputs = model(**inputs).logits
#             generated_ids = model.generate(**inputs)

#             # trimmed_generated_ids = [
#             #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
#             # ]
#             print(labels.shape, generated_ids.shape,inputs['input_ids'].shape)
#             # trimmed_generated_ids = generated_ids[:,labels.shape[1]:]

#             # Decode the output text
#             output_text = processor.batch_decode(
#                 generated_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False
#             )

#             input_text = processor.batch_decode(
#                 inputs['input_ids'],
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False
#             )
#             print("Input",input_text)
#             print("Output",output_text)

#             # decoded_preds = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
#             # print(labels.shape,outputs.shape)
#             preds = outputs.argmax(dim=-1)
#             # print(preds.shape,labels.size(0))

#         total_correct += (preds == labels).sum().item()
        
#         total_samples += labels.size(0)*labels.size(1)
#         # print("Total",total_correct,' out of',labels.size(0)*labels.size(1))

#     accuracy = total_correct / total_samples
#     return accuracy
def get_prompt(batch, processor):

    reference_texts = processor.batch_decode(
        batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # prompt_only_text 
    # print(reference_texts)
def evaluate_model(model, config, dataset, processor, preprocess_fn=None, batch_size=8, device="cuda"):
    if config['models'][0]['name'] == 'qwen':
        evaluate_model_qwen(model, config, dataset, processor, preprocess_fn, batch_size, device)
    elif config['models'][0]['name'] == 'florence': 
        evaluate_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, device)
    elif config['models'][0]['name'] == 'florence-large': 
        evaluate_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, device)

def evaluate_model_florence(model, config, dataset, processor, preprocess_fn=None, batch_size=8, device="cuda"):
    total_val_loss = 0  

    # Validation loop
    val_loss = 0
    predicted_answers = []
    ground_truth_answers = []
    task_prompt = config["task_prompt"] 

    loader = dataset['val']
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # model.eval()

        # EVALUATE THE MODEL USING LEVENSHTEIN SIMILARITY
        # j = 0
        for inputs, answers in tqdm(loader, desc=f"Validation Epoch"):
            labels = processor.tokenizer(
                text=answers, 
                return_tensors="pt", 
                padding=True, 
                return_token_type_ids=False
            ).input_ids.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Here Forward pass until you get the result from the florence VLM decoder
            outputs = model(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], labels=labels )
            loss1 = outputs.loss # CrossEntropy Loss
            val_loss += loss1.item()

            # Generate output from the model for Levenshtein similarity calculation
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

            # Decode generated text
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            # Process and collect answers for Levenshtein similarity
            for generated_text, gt_answer in zip(generated_texts, answers):
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(
                        inputs["pixel_values"].shape[-2],
                        inputs["pixel_values"].shape[-1],
                    ),
                )
                
                # Extract and clean the answer
                pred_answer = parsed_answer[task_prompt].replace("<pad>", "").strip()
                predicted_answers.append(pred_answer)
                ground_truth_answers.append(gt_answer)
            # break
        # Calculate metrics 
        avg_val_loss = val_loss / len(loader)
        levenshtein_similarity = average_normalized_levenshtein_similarity(
            ground_truth_answers, predicted_answers
        )

        # Print metrics 
        print(f" Validation Loss: {avg_val_loss}")
        print(f"Levenshtein Similarity: {levenshtein_similarity:.4f}")

        
def evaluate_model_qwen(model, dataset, processor, preprocess_fn=None, batch_size=8, device="cuda"):


    loader = dataset['val']
    model.eval()
    model.to(device)

    total_correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluating"):
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        labels = batch["labels"].to(device)
        refs = inputs.pop('ref')
        # del batch['ref']

        # get_prompt(batch,processor)

        # Generate responses
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = processor.tokenizer.pad_token_id

        # Decode generated output and label
        generated_texts = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        reference_texts = processor.batch_decode(
            safe_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for pred, ref in zip(generated_texts, refs):
            # print('pred',pred)
            if "assistant\n" in pred:
                pred = pred.split("assistant\n")[1]
            if "<|im_end|>" in ref:
                ref = ref.split("<|im_end|>")[0]
            # print('pred', pred)
            # print('ref',ref)
            pred = pred.strip()
            ref = ref.strip()
            if pred == ref:
                total_correct += 1
            total_samples += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy