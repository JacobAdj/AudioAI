import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the smallest SeamlessM4T model
model_name = "facebook/hf-seamless-m4t-medium"


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

cacheHF = "D:/LanguageModels/hf/"

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/seamless-m4t-v2-large", cache_dir=cacheHF)
tokenizer = AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large", cache_dir=cacheHF)


model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text for TTS
text = "Hello, how are you?"

# Tokenize text
inputs = tokenizer(text, return_tensors="pt")

# Generate speech
output = model.generate(**inputs)
translated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Speech Text:", translated_text)
