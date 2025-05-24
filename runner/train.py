import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from runner.evaluate import evaluate_model
from tqdm import tqdm

def freeze_model_except_adapters(model):
    for name, param in model.named_parameters():
        # print("Adapter? ",name)
        # if "Sequential" in name or "adapter" or "model.layers.27.input_layernorm" in name:
        if "Sequential" in name or "adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # print number of trainable parameters here.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    all_params = sum(p.numel() for p in model.parameters())

    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {trainable_params/all_params*100}")

def train_model(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    if config['models'][0]['name'] == 'qwen':
        train_model_qwen(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'florence': 
        train_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'florence-large': 
        train_model_florence(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)
    elif config['models'][0]['name'] == 'mistral-7b-instruct':
        train_model_mistral(model, config, dataset, processor, preprocess_fn, batch_size, epochs, device, lr)

def train_model_mistral(original_model, tokenizer, train_dataset, args):
    if isinstance(train_dataset, dict) and 'train' in train_dataset:
        data = train_dataset['train']
    else:
        data = train_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model.to(device)
    original_model.train()

    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, original_model.parameters()), lr=args.learning_rate)

    batch_size = args.per_device_train_batch_size
    epochs = int(args.num_train_epochs)

    for epoch in range(epochs):
        total_loss = 0.0
        # Ensure data is a list for slicing and len()
        if not isinstance(data, list):
            data = list(data) # Convert to list if it's a Hugging Face Dataset object for example

        pbar = tqdm(range(0, len(data), batch_size), desc=f"Mistral Training Epoch {epoch+1}/{epochs}")
        for i in pbar:
            batch_samples = data[i:i+batch_size]
            if not batch_samples:
                continue

            # Prepare input and label tensors
            # Assuming the dataset format from main_mistral_injection.py context
            # where each item is a dictionary with a "history" key.
            try:
                inputs_text = [s["history"][-1]["user"] for s in batch_samples]
                labels_text = [s["history"][-1]["bot"] for s in batch_samples]
            except (KeyError, IndexError, TypeError) as e:
                # Handle cases where data might not be in the expected format
                # This could happen if train_dataset is not a list of dicts as expected
                # or if "history" or its elements are missing.
                # For now, we'll skip such batches with a warning.
                print(f"Warning: Skipping batch due to unexpected data format: {e}. Batch sample: {batch_samples[0] if batch_samples else 'empty'}")
                continue


            # Tokenize
            # The tokenizer should handle padding and truncation.
            # Ensure tokenizer.pad_token is set if not already.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # For Mistral, typically the input includes both user and bot utterances for Causal LM.
            # The labels are then shifted versions of the input_ids.
            # However, the current structure suggests separate inputs and labels.
            # Let's assume the task is to predict `labels_text` given `inputs_text`.
            # For Causal LM, we often concatenate them: prompt + completion
            # And then the labels are the input_ids, with padding/prompt tokens masked.

            # Let's adapt to a common instruction-following format:
            # input: "USER: user_query BOT:"
            # label: "bot_response <eos>"
            # The model then predicts the bot_response.
            # The loss is calculated only on the bot_response part.

            full_texts = [f"USER: {s['history'][-1]['user']} BOT: {s['history'][-1]['bot']}{tokenizer.eos_token}" for s in batch_samples]
            
            tokenized_outputs = tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = tokenized_outputs.input_ids.to(device)
            attention_mask = tokenized_outputs.attention_mask.to(device)
            labels = input_ids.clone() # Labels are initially the same as input_ids

            # Mask tokens from the "USER: user_query BOT:" part for loss calculation
            # We only want to calculate loss on the bot's response part.
            for idx, sample in enumerate(batch_samples):
                user_prompt_part = f"USER: {sample['history'][-1]['user']} BOT:"
                tokenized_user_prompt = tokenizer(user_prompt_part, return_tensors="pt", add_special_tokens=False)
                prompt_len = tokenized_user_prompt.input_ids.shape[1]
                labels[idx, :prompt_len] = -100 # -100 is the ignore_index for CrossEntropyLoss

            # Forward pass
            outputs = original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{total_loss/( (i//batch_size) + 1):.4f}"})

        avg_epoch_loss = total_loss / (len(data) // batch_size + 1) if (len(data) // batch_size + 1) > 0 else total_loss
        print(f"[Epoch {epoch+1}/{epochs}] Avg Loss: {avg_epoch_loss:.4f}")
    return original_model

def train_model_florence(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    loader = dataset['train']
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(epochs):  
        # TODO: Adapter train mode
        # linear_layer.train()
        total_train_loss = 0  

        for idx,batch in enumerate(loader):
            # break
            inputs, answers = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            image_features = model._encode_image(inputs['pixel_values']).to(device)
            input_embeds = model.get_input_embeddings()(inputs['input_ids']).to(device)
            input_embeds, attention_mask = model._merge_input_ids_with_image_features(input_embeds, image_features)

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"] 
            labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to('cuda')
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)

            # Compute loss
            # loss = loss_criterion(outputs.logits.view(-1, outputs.logits.size(-1)),labels.view(-1))
            loss = outputs.loss
            # print('loss',loss)
            # print("NAN: ",torch.isnan(outputs.logits).any())
            # print("NAN: ",outputs.logits)  # Should be False
            # print("NAN: ",torch.isnan(labels).any())  # Should be False
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()  
            # wandb.log({
            # "train_loss": loss.item(),
            # "batch_id": idx
            # })
            if idx == 50:
                break
        avg_train_loss = total_train_loss / 50 
        print(avg_train_loss)

        # Validation Loop
        # Adapter validation 
        # linear_layer.eval()
        if epoch % 20 == 0:
            evaluate_model(model, config, dataset,processor)
            # print(f"Accuracy: {acc:.4f}")



def train_model_qwen(model, config, dataset, processor, preprocess_fn=None, batch_size=8, epochs=3, device="cuda", lr=0.0001):
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loader = dataset['train']
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        idx =1
        pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}")
        for batch in pbar:
            inputs = preprocess_fn(batch) if preprocess_fn else batch
            labels = batch["labels"].to(device)
            # print(labels.shape)

            inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            # refs = inputs.pop('ref')
            outputs = model(**inputs)
            loss = outputs.loss
            # print(outputs)
            # loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Loss: {total_loss/idx:.4f}")
            idx+=1

            if idx==50:
                break

        if epoch%20 == 0:
            acc = evaluate_model(model, config, dataset,processor)
            print(f"Accuracy: {acc:.4f}")


        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(loader):.4f}")