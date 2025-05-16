import yaml
import torch
# from adapter.my_adapter import DCTAdapter #FrequencyGatedDCTAdapter
from adapter.my_adapter import DCTAdapter
from models.vlm_loader import load_vlm
from dataset.dataset_loader import load_dataset_vlm
from runner.evaluate import evaluate_model
from runner.train import train_model, freeze_model_except_adapters
from peft import LoraConfig, get_peft_model, IA3Config
from peft import LoHaConfig,AdaLoraConfig, LoKrConfig

def inject_adapters(model, adapter_cls, adapter_args,layers):
    for name, module in model.named_modules():
        print(name)
        # print(layers)
        # if "model.layers.27.input_layernorm" in name:
        for layer in layers:
            
            if layer['name'] in name:
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
        print("Model ", model_cfg["name"])
        model = load_vlm(model_cfg["name"])
        if config.get("adapter", {}).get("do_adapt", False):
            print("Injecting Adapters.....")
            model = inject_adapters(model, DCTAdapter, config["adapter"]["params"], config["adapter"]["layers"])
            freeze_model_except_adapters(model)
        if config.get("adapter", {}).get("do_peft", False):
            print('Do PEFT.....')
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=3,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            )
            # peft_config = IA3Config(
            #     task_type="CAUSAL_LM", 
            #     target_modules=["k_proj", "v_proj", "down_proj"], 
            #     feedforward_modules=["down_proj"]
            # )
            
            # peft_config= LoHaConfig(
            #     r=16,
            #     alpha=16,
            #     target_modules=["q_proj", "v_proj"],
            #     module_dropout=0.1,
            #     modules_to_save=["classifier"],
            #     task_type="CAUSAL_LM"
            # )

            # peft_config = AdaLoraConfig(
            #     r=8,
            #     init_r=12,
            #     tinit=200,
            #     tfinal=1000,
            #     deltaT=10,
            #     total_step=10000,
            #     target_modules=["q_proj", "v_proj"],
            #     task_type="CAUSAL_LM",
            # )
            # peft_config = LoKrConfig(
            #     r=16,
            #     alpha=16,
            #     target_modules=["q_proj", "v_proj"],
            #     module_dropout=0.1,
            #     task_type="CAUSAL_LM",
            #     # modules_to_save=["classifier"],
            # )

            # Apply PEFT model adaptation
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        for dataset_cfg in config["datasets"]:
            dataset,processor = load_dataset_vlm(dataset_cfg["name"])

            if config.get("train", {}).get("do_train", False):
                print(f"Training on {dataset_cfg['name']}...")
                # 
                train_model(
                    model,
                    config,
                    dataset,
                    processor,
                    batch_size=config["train"]["batch_size"],
                    epochs=config["train"]["epochs"],
                    lr=config["train"]["lr"],
                )

            print(f"Evaluating on {dataset_cfg['name']}...")
            acc = evaluate_model(model,config, dataset,processor)
            print("accuracy ", acc)
            print("ALLL_DONE!!!!")
            print(f"Model: {model_cfg['name']}, Dataset: {dataset_cfg['name']}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()