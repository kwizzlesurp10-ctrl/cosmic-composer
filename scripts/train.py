from models.transformer import CosmicTransformer
from data.loader import AudioTextDataset
import torch

def train_model(dataset, epochs, batch_size):
    model = CosmicTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for batch in dataloader:
            audio, text, neural = batch
            loss = model.train_step(audio, text, neural)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    torch.save(model.state_dict(), os.environ.get('MODEL_CHECKPOINT', 'models/checkpoint.pth'))
