import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

# Step 1: Load Fine-Tuned Model and Tokenizer
model_name = "./fine_tuned_phi3_new" # <- Local Model "microsoft/phi2" <- Remote Model 

# Load the fine-tuned model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure consistency

# Step 2: Define Input Prompts
test_prompts = [
   "Once upon a day"
]

# Step 3: Generate Predictions for Each Prompt
print("Generating text for test prompts...\n")
for prompt in test_prompts:
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        inputs["input_ids"],
        max_length=1000,  # Maximum length of the generated output
        temperature=1,  # Sampling temperature (lower values = more deterministic)
        top_k=50,  # Top-k sampling
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.2,  # Penalize repetitive text
        num_return_sequences=1,  # Generate one completion per prompt
    )

    # Decode and display the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generated_text}")
    print("-" * 50)

# Step 4: Evaluate Using Perplexity (Optional)
def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    labels = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()


# Example text for perplexity evaluation
test_text = "This is a simple example to evaluate perplexity."
perplexity = calculate_perplexity(model, tokenizer, test_text)
print(f"Perplexity for '{test_text}': {perplexity}")
