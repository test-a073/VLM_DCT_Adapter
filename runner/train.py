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