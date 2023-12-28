import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup

class CustomDataset(Dataset):
    def __init__(self, tokenizer, filename, block_size=512):
        with open(filename, 'r', encoding='utf8') as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size, padding='max_length', truncation=True)["input_ids"]
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

def fine_tune(model_name, train_file, epochs, batch_size, learning_rate, save_path):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)  # Move the model to the GPU
    
    # Load dataset
    dataset = CustomDataset(tokenizer, train_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * epochs)

    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            inputs = batch.to(device)  # Move the batch to the GPU
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

            # Clearing the cache after each batch
            torch.cuda.empty_cache()

    # Save the model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model

# Training:

save_path = "/home/ali0rez/Documents/gpt2_model_output"
model = fine_tune("gpt2-medium", "/home/ali0rez/Documents/data.txt", epochs=40, batch_size=20, learning_rate=5e-5, save_path=save_path)


