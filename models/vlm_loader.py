from transformers import Blip2ForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

def load_vlm(model_name):
    if model_name == "qwen":
        model_id = "Qwen/Qwen2-VL-7B-Instruct"

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            # device_map="auto",
            # torch_dtype=torch.bfloat32,
        )
        return model

    elif model_name == "blip2":
        return Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    elif model_name == "florence":

        florence_base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base-ft",
            trust_remote_code=True,
            # revision=config.BASE_MODEL_REVISION, 
            # torch_dtype=config.TORCH_DTYPE,
            attn_implementation="eager"
        )
    elif model_name == "florence-large":

        florence_base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large-ft",
            trust_remote_code=True,
            # revision=config.BASE_MODEL_REVISION, 
            # torch_dtype=config.TORCH_DTYPE,
            attn_implementation="eager"
        )
        return florence_base_model

    else:
        raise ValueError(f"Unknown model {model_name}")