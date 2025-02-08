import os
import torch
import random  # For random sampling
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM,
    DataCollatorForSeq2Seq, BitsAndBytesConfig, AutoConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import chardet

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Path Configuration
CACHE_PATH = "./cached_dataset_phi3_new2"
MODEL_NAME = "./fine_tuned_phi3"
OUTPUT_DIR = "./fine_tuned_phi3_new"
DATA_FOLDER = "./StoriesAll"

# Training Hyperparameters
# Training Hyperparameters for 1 Epoch
MAX_LENGTH_C = 2048  # Keep max token length
BATCH_SIZE = 12  # Increase batch size for faster training
LEARNING_RATE = 1.5e-4  # Slightly higher LR since only 1 epoch
GRAD_ACCUM_STEPS = 2  # Maintain stability
EPOCHS = 1  # Only 1 epoch


# Function to Detect and Convert File Encoding
def detect_and_convert_to_utf8(file_path):
    """Detect file encoding and convert to UTF-8."""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        encoding = chardet.detect(raw_data)['encoding']
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


# Function to Load All Stories
def load_stories(folder_path, num_samples=700):
    """Load all stories, shuffle, and select a random subset (num_samples)."""
    stories = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    stories.append(detect_and_convert_to_utf8(file_path))
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Shuffle stories and select a random subset
    random.shuffle(stories)
    return stories[:num_samples]


# Function to Chunk Stories into Overlapping Segments
def chunk_stories(stories, max_length_c, overlap=0.5):
    """Splits each story into overlapping chunks of max_length_c with overlap."""
    chunked_stories = []
    stride = int(max_length_c * (1 - overlap))  # Step size for overlapping

    for story in stories:
        chunks = []
        for i in range(0, len(story) - max_length_c + 1, stride):
            chunk = story[i:i + max_length_c]
            chunks.append(chunk)

        # Add the last chunk if it's shorter than max_length_c
        if len(story) % stride != 0:
            chunks.append(story[-max_length_c:])

        chunked_stories.extend(chunks)

    return chunked_stories


# Tokenization Function
def tokenize_function(examples):
    """Tokenize each chunk with truncation and padding, and prepare labels for causal LM."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH_C,
    )
    tokenized["label"] = tokenized["input_ids"].copy()
    return tokenized


# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load or Process Dataset
if os.path.exists(CACHE_PATH):
    print("Loading dataset from cache...")
    tokenized_datasets = load_from_disk(CACHE_PATH)
else:
    print("Loading and processing stories...")
    all_stories = load_stories(DATA_FOLDER)
    print(f"Loaded {len(all_stories)} stories")

    # Apply chunking before tokenization
    chunked_texts = chunk_stories(all_stories, MAX_LENGTH_C)

    # Create dataset with chunked texts
    chunked_dataset = Dataset.from_dict({"text": chunked_texts})
    print(f"Chunked dataset generated with {len(chunked_texts)} chunks")

    # Tokenize and save dataset
    tokenized_datasets = chunked_dataset.map(tokenize_function, batched=True)
    tokenized_datasets.save_to_disk(CACHE_PATH)
    print(f"Tokenized dataset saved to {CACHE_PATH}")

# Split Dataset into Training and Evaluation
split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]
print(f"Training Dataset Size: {len(train_dataset)}")
print(f"Evaluation Dataset Size: {len(eval_dataset)}")


try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enables 4-bit loading
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Double quantization for memory efficiency
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,  # Apply quantization
        use_cache=False,
        return_dict=True
    )
    print("Base Model loaded successfully!")
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    print(f"Model's maximum sequence length: {model_config.max_position_embeddings}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# Prepare Model for LoRA Fine-Tuning
base_model = prepare_model_for_kbit_training(base_model)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj"
    ],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM",
)


# Apply LoRA to Model
model = get_peft_model(base_model, lora_config)
print("LoRA Adapters Added")

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results_phi3",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    save_strategy="steps",
    save_steps=200,
    fp16=True,
    logging_dir="./logs_phi3",
    logging_steps=25,
    save_total_limit=2,
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    padding=True,
    max_length=MAX_LENGTH_C,
    return_tensors="pt",
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Start Training
print("Starting Training...")
checkpoint_path = "./results_phi3/checkpoint-1600"
if os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("Checkpoint not found. Starting training from scratch.")
    trainer.train()

print("Training Complete")

# Save Fine-Tuned Model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model Saved")
