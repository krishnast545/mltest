# Import necessary libraries
import os
import mlflow
import mlflow.pytorch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Clear any MLflow environment variables that might hold onto a previous run ID
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://192.168.74.161:8080")

# Specify the name of the existing experiment
experiment_name = "lfi1"  # Replace with the name of your experiment
mlflow.set_experiment(experiment_name)

# Initialize the tokenizer and model
model_name = "gpt2"  # A small, pre-trained model for conversation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# End any existing MLflow run at the beginning of the script
if mlflow.active_run():
    mlflow.end_run()

# Log the model configuration and parameters to MLflow
with mlflow.start_run():
    mlflow.log_param("model_name", model_name)

    # Log the unmodified pre-trained model to MLflow
    mlflow.pytorch.log_model(model, artifact_path="simple_pretrained_model")

    print("Model logged to MLflow under experiment:", experiment_name)

# End the MLflow run explicitly to prevent any conflict with further runs
mlflow.end_run()

# Real-time conversation function
def generate_response(prompt):
    # Tokenize the input prompt and create attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate output with attention mask
    output = model.generate(
        inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),  # Get attention mask if available
        max_length=150,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Real-time conversation loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    prompt = input("User: ")
    if prompt.lower() == "exit":
        print("Conversation ended.")
        break
    response = generate_response(prompt)
    print("Bot:", response)
