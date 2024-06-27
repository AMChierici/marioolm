import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Dataset class
class ItalianLyricsDataset(Dataset):
    def __init__(self, lyrics, tokenizer, max_length):
        self.lyrics = lyrics
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        encoding = self.tokenizer(lyric, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()

# RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)

# Function to train models
def train_model(model, dataloader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, masks = batch
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Main function
def main():
    # Load and preprocess data
    with open('./data/italian_lyrics.txt', 'r', encoding='utf-8') as f:
        lyrics = f.readlines()
    
    # Initialize tokenizer and models
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    hidden_dim = 512
    max_length = 128
    batch_size = 32
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    dataset = ItalianLyricsDataset(lyrics, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim).to(device)
    lstm_model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
    transformer_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

    # Train models
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters())
    lstm_optimizer = optim.Adam(lstm_model.parameters())
    transformer_optimizer = optim.Adam(transformer_model.parameters())

    print("Training RNN model...")
    train_model(rnn_model, dataloader, criterion, rnn_optimizer, device, epochs)

    print("Training LSTM model...")
    train_model(lstm_model, dataloader, criterion, lstm_optimizer, device, epochs)

    print("Training Transformer model...")
    train_model(transformer_model, dataloader, criterion, transformer_optimizer, device, epochs)

    # Generate sample lyrics
    def generate_lyrics(model, tokenizer, seed_text, max_length=50):
        model.eval()
        input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    print("\nGenerating sample lyrics:")
    seed_text = "Amore mio"
    print("RNN:", generate_lyrics(rnn_model, tokenizer, seed_text))
    print("LSTM:", generate_lyrics(lstm_model, tokenizer, seed_text))
    print("Transformer:", generate_lyrics(transformer_model, tokenizer, seed_text))

if __name__ == "__main__":
    main()