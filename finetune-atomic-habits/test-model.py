from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel

# Define the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    torch_dtype=torch.float16
)

# Load the model with the quantization config
base_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    quantization_config=quantization_config
)

model = PeftModel.from_pretrained(base_model, "./Final Adapter")

tokenizer = AutoTokenizer.from_pretrained("./Final Adapter")


# Test the model with a question
input_text = "Question: Who are you? How do I build a habit?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))