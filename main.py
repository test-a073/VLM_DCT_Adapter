import yaml
import torch
from adapter.my_adapter import MyCustomAdapter
from models.vlm_loader import load_vlm
from dataset.dataset_loader import load_dataset_vlm
from runner.evaluate import evaluate_model
from runner.train import train_model, freeze_model_except_adapters

def inject_adapters(model, adapter_cls, adapter_args):
    for name, module in model.named_modules():
        if "layer.11.attention.output.dropout" in name:
            # print('Injecting:', name)
            parent = get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], torch.nn.Sequential(module, adapter_cls(**adapter_args)))
        # if isinstance(module, torch.nn.Linear):
        #     parent = get_parent_module(model, name)
        #     setattr(parent, name.split('.')[-1], torch.nn.Sequential(module, adapter_cls(**adapter_args)))
    return model

def get_parent_module(model, name):
    names = name.split('.')
    for n in names[:-1]:
        model = getattr(model, n)
    return model

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for model_cfg in config["models"]:
        model = load_vlm(model_cfg["name"])
        model = inject_adapters(model, MyCustomAdapter, config["adapter"]["params"])
        
        for dataset_cfg in config["datasets"]:
            dataset = load_dataset_vlm(dataset_cfg["name"])

            if config.get("train", {}).get("do_train", False):
                print(f"Training on {dataset_cfg['name']}...")
                freeze_model_except_adapters(model)
                train_model(
                    model,
                    dataset,
                    batch_size=config["train"]["batch_size"],
                    epochs=config["train"]["epochs"],
                    lr=config["train"]["lr"],
                )

            print(f"Evaluating on {dataset_cfg['name']}...")
            acc = evaluate_model(model, dataset)
            print(f"Model: {model_cfg['name']}, Dataset: {dataset_cfg['name']}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()