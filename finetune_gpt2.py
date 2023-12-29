import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from datetime import datetime

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

    # Use the AutoTokenizer and AutoModelForCausalLM for Mixtral
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])  # Utilize multiple GPUs

    # Load dataset
    dataset = CustomDataset(tokenizer, train_file)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * epochs)

    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            progress_bar.set_postfix(loss=loss.item(), timestamp=timestamp)

            torch.cuda.empty_cache()

    # Save the model and tokenizer
    model.module.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model

# Training:
save_path = "/home/ali0rez/Documents/mixtral_model_output"
model_name = "mistralai/Mixtral-8x7B-v0.1"
train_file = "/home/ali0rez/Documents/data.txt"
model = fine_tune(model_name, train_file, epochs=1, batch_size=20, learning_rate=1e-5, save_path=save_path)
