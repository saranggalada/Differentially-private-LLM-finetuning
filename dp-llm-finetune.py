## Import libraries
# For parsing configs and args
import argparse
import json
import os
# For standard ML procedures
import numpy as np
import torch
from torch.utils.data import DataLoader
# For loading readymade models and datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
# For parameter-efficient training
import peft
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
# For differential privacy
import opacus
from opacus import PrivacyEngine  # For differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager  # For large batch sizes
# For monitoring training progress
from tqdm import tqdm
# For accessing HuggingFace Hub
from huggingface_hub import login, HfApi, HfFolder, Repository


def load_config(config_path):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Perform type conversions and calculations after loading
            config['MAX_TOKENS'] = int(config.get('MAX_TOKENS', 512)) #provide default value if not in json
            config['GRADIENT_ACCUMULATION_STEPS'] = int(config.get('GRADIENT_ACCUMULATION_STEPS', 4))
            config['SANITY_CHECK_SIZE'] = int(config.get('SANITY_CHECK_SIZE', 1000))
            config['MAX_PHYSICAL_BATCH_SIZE'] = int(config.get('MAX_PHYSICAL_BATCH_SIZE', 8))
            config['USE_SANITY_CHECK'] = bool(config.get('USE_SANITY_CHECK', True))
            config['NUM_EPOCHS'] = int(config.get('NUM_EPOCHS', 3))
            config['BATCH_SIZE'] = int(config.get('BATCH_SIZE', 8))
            config['LEARNING_RATE'] = float(config.get('LEARNING_RATE', 5e-5))
            config['EPSILON'] = float(config.get('EPSILON', 7.5))
            config['DELTA'] = float(config.get('DELTA', 1 / config['SANITY_CHECK_SIZE'])) if config.get('SANITY_CHECK_SIZE') is not None else 1/1000 #calculate delta after sanity check size is loaded
            config['MAX_GRAD_NORM'] = float(config.get('MAX_GRAD_NORM', 1.0))
            return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}. Creating default config file.")
        default_config = {
            "MODEL_NAME": "",
            "DATASET_NAME": "",
            "TEXT_COLUMN": "",
            "MAX_TOKENS": 512,
            "GRADIENT_ACCUMULATION_STEPS": 4,
            "SANITY_CHECK_SIZE": 1000,
            "MAX_PHYSICAL_BATCH_SIZE": 8,
            "USE_SANITY_CHECK": True,
            "NUM_EPOCHS": 3,
            "BATCH_SIZE": 8,
            "LEARNING_RATE": 5e-5,
            "EPSILON": 7.5,
            "DELTA": 1/1000,
            "MAX_GRAD_NORM": 1.0,
            "HF_READ_TOKEN": "",
            "HF_WRITE_TOKEN": "",
            "SAVE_MODEL_REPO_NAME": ""
        }
        with open(config_path, 'w') as outfile:
            json.dump(default_config, outfile, indent=4)
        return default_config
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {config_path}. Please check the file.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model.")
    parser.add_argument("--config", required=True, help="Path to the config file.")
    args = parser.parse_args()

    config = load_config(args.config)

    if config is None:
        print("Config loading failed, exiting")
        exit(1)

    print("Loaded configuration:")
    print(json.dumps(config, indent=4))

    # Access config parameters:
    MODEL_NAME = config["MODEL_NAME"]
    DATASET_NAME = config["DATASET_NAME"]
    TEXT_COLUMN = config["TEXT_COLUMN"]
    MAX_TOKENS = config["MAX_TOKENS"]
    GRADIENT_ACCUMULATION_STEPS = config["GRADIENT_ACCUMULATION_STEPS"]
    USE_SANITY_CHECK = config["USE_SANITY_CHECK"]
    SANITY_CHECK_SIZE = config["SANITY_CHECK_SIZE"]
    MAX_PHYSICAL_BATCH_SIZE = config["MAX_PHYSICAL_BATCH_SIZE"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    LEARNING_RATE = config["LEARNING_RATE"]
    EPSILON = config["EPSILON"]
    DELTA = config["DELTA"]
    MAX_GRAD_NORM = config["MAX_GRAD_NORM"]
    HF_READ_TOKEN = config["HF_READ_TOKEN"]
    HF_WRITE_TOKEN = config["HF_WRITE_TOKEN"]
    SAVE_MODEL_REPO_NAME = config["SAVE_MODEL_REPO_NAME"]


    # --- START OF FINE-TUNING CODE ---
    try:
        # Your fine-tuning logic goes here. Use the config parameters.
        print("Starting Fine-tuning")
        print(f"Fine-tuning {MODEL_NAME} on {DATASET_NAME}...")

        ### START Core fine-tuning code ###
        

        # 1. HF login
        # HF_READ_TOKEN = "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" # Sarang's HF read Token
        login(HF_READ_TOKEN)

        # 2. Configure device - Use a GPU if available
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. Load dataset
        # certain datasets contain different versions (aka splits), which need to be specified
        if DATASET_NAME == "wikitext":
            DATASET_SPLIT = "wikitext-2-raw-v1"
            data = load_dataset(DATASET_NAME, DATASET_SPLIT) ### Only required by certain datasets (eg. wikitexts)
        else:
            data = load_dataset(DATASET_NAME)

        # Select train and test splits for the text column of interest
        train_texts = data["train"][TEXT_COLUMN]
        if 'test' in data.keys():
            eval_texts = data["test"][TEXT_COLUMN]   
            
        if USE_SANITY_CHECK: # If simply sanity-checking, select a very small subset for training
            train_texts = train_texts[:SANITY_CHECK_SIZE]
            if 'test' in data.keys():
                eval_texts = eval_texts[:SANITY_CHECK_SIZE]

        # By default, set Delta to reciprocal of training set size
        DELTA = 1 / len(train_texts)

        # 4. Load Model
        # Load pretrained model configurations and tokenizer
        config = AutoConfig.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Create a quantization configuration for 4-bit precision
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        # Load pretrained model based on configurations
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            config=config,
            quantization_config=quantization_config, 
            # device_map="auto"
        )

        # Prepare model for low-bit training - essential to make the LoRA we apply later "quantization aware"
        model = prepare_model_for_kbit_training(model)

        # Ensure padding token is set as EOS
        tokenizer.pad_token = tokenizer.eos_token


        # 5. Apply QLoRA - LoRA configuration specific to quantized models
        # For Llama models, include target modules such as q_proj, v_proj
        if "llama" in MODEL_NAME:
            lora_config = LoraConfig(
                r=8, # Rank of the low-rank matrices
                lora_alpha=32, # Scaling factor for the LoRA updates
                target_modules=["q_proj", "v_proj"], # Modules to apply LoRA to  ### Modify as per model architecture
                lora_dropout=0.05, # Dropout probability applied to the LoRA updates for regularization
                bias="none", # Whether to include bias parameters in the LoRA layers
                task_type="CAUSAL_LM" # Type of task - eg. causal modelling, seq2seq
            )
        else:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

        # Obtain the parameter-efficient LoRA model
        model = get_peft_model(model, lora_config)


        # 6. Tokenization function
        def tokenize_function(examples):
            return tokenizer(examples, padding="max_length", truncation=True, max_length=MAX_TOKENS)

        # Tokenize each training sample - numeric representation of the text ("input_ids")
        tokenized_train_texts = [tokenize_function(text) for text in train_texts]
        if 'test' in data.keys():
            tokenized_eval_texts = [tokenize_function(text) for text in eval_texts]

        # Prepare train_loader and eval_loader
        train_dataset = [(torch.tensor(t["input_ids"]), torch.tensor(t["attention_mask"])) for t in tokenized_train_texts]
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        if 'test' in data.keys():
            eval_dataset = [(torch.tensor(t["input_ids"]), torch.tensor(t["attention_mask"])) for t in tokenized_eval_texts]
            eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)


        # 7. Wrap the model, optimizer, dataloader with Opacus' DP engine
        # This introduces per-sample gradient calculation, noise injection and gradient clipping
        model = model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        privacy_engine = PrivacyEngine() # secure_mode=True requires torchcsprng to be installed

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=NUM_EPOCHS,
            target_delta=DELTA,  # Privacy budget
            target_epsilon=EPSILON,  # Probability of privacy breach
            max_grad_norm=MAX_GRAD_NORM, # threshold for clipping the norm of per-sample gradients
        )


        # 8. Training loop with BatchMemoryManager
        for epoch in range(1, NUM_EPOCHS + 1):
            losses = []
            model.train()

            # Use BatchMemoryManager for managing memory
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                optimizer=optimizer
            ) as memory_safe_loader:

                # Training step
                for step, batch in enumerate(tqdm(memory_safe_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
                    optimizer.zero_grad()

                    # Move batch to DEVICE
                    input_ids, attention_mask = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    labels = input_ids.clone()  # Labels are the same as input_ids for causal LM

                    # Forward pass - loss calculation
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    losses.append(loss.item())

                    # Backward pass - parameter update
                    loss.backward()
                    optimizer.step()

                    # Log progress every 50 steps
                    if step > 0 and step % 50 == 0:
                        train_loss = np.mean(losses)
                        epsilon = privacy_engine.get_epsilon(DELTA)

                        print(
                            f"Epoch: {epoch} | Step: {step} | "
                            f"Train loss: {train_loss:.3f} | "
                            f"ɛ: {epsilon:.2f}"
                        )

            # Epoch summary
            train_loss = np.mean(losses)
            epsilon = privacy_engine.get_epsilon(DELTA)
            print(f"Epoch {epoch} completed. Average loss: {train_loss:.4f}, ɛ: {epsilon:.2f}")


        # 9. Unwrap the DP fine-tuned model - our model is wrapped by a PEFT wrapper and Opacus wrapper
        ## Step 1: Check if the model is wrapped in GradSampleModule (Opacus wrapper)
        if isinstance(model, opacus.grad_sample.GradSampleModule):
            unwrapped_model = model._module  # Access the underlying model from GradSampleModule
        else:
            unwrapped_model = model  # If not wrapped, use the model as-is
        ## Step 2: For LoRA/PEFT models, unwrap further if needed
        if isinstance(unwrapped_model, peft.PeftModelForCausalLM):
            inference_model = unwrapped_model.base_model  # Extract the base LoRA model
        else:
            inference_model = unwrapped_model  # If not a PEFT model, use as-is
        # Set model for inference by freezing parameters
        inference_model.eval()
        print("model unwrapped!")


        # 10. Push to HuggingFace Hub
        # HuggingFace 'write' token
        inference_model.push_to_hub(SAVE_MODEL_REPO_NAME, token=HF_WRITE_TOKEN)
        tokenizer.push_to_hub(SAVE_MODEL_REPO_NAME, token=HF_WRITE_TOKEN)
        print(f"Model and tokenizer pushed to Hugging Face Hub: https://huggingface.co/{SAVE_MODEL_REPO_NAME}")


        ### END Core fine-tuning code ###

        print("Fine-tuning complete.")
    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")
        exit(1)
    # --- END OF FINE-TUNING CODE ---

if __name__ == "__main__":
    main()