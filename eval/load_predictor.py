import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from mesh.dataset.utils import MODEL_SAVE_DIR

PREDICTOR_DIR = os.path.join(MODEL_SAVE_DIR, 'predictor')
if not os.path.exists(PREDICTOR_DIR):
    os.makedirs(PREDICTOR_DIR)


def generate_seq(seq_len, num_samples):
    main = torch.sin(torch.linspace(0, 7, seq_len *
                     num_samples)).view(num_samples, seq_len)
    nearby = torch.cos(torch.linspace(0, 8, seq_len *
                       num_samples)).view(num_samples, seq_len)
    return main, nearby


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerPredictor, self).__init__()
        self.lr = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        src = self.lr(src)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = self.fc(output).squeeze(-1)
        return output

    def save_model(self, epoch):
        model_path = os.path.join(
            PREDICTOR_DIR, f'predictor_epoch_{epoch}.pth')
        torch.save(self.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    def load_model(self, epoch):
        model_path = os.path.join(
            PREDICTOR_DIR, f'predictor_epoch_{epoch}.pth')
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, weights_only=True))
            print(f'Model loaded from {model_path}')
        else:
            print(f'No model found at {model_path}')

num_epochs = 50

if __name__ == "__main__":
    
    seq_len = 100
    num_samples = 1000
    his_t = 50
    pred_t = seq_len - his_t
    min_len = min(pred_t, his_t)

    main, nearby = generate_seq(seq_len, num_samples)
    # Shape: (num_samples, seq_len, 2)
    data = torch.stack((main, nearby), dim=-1)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    train_data = TensorDataset(train_data.dataset[train_data.indices])
    test_data = TensorDataset(test_data.dataset[test_data.indices])
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = TransformerPredictor(
        input_dim=2, model_dim=64, num_heads=4, num_layers=2, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5)


    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            src = batch[0][:, :his_t, :]  # (batch_size, his_t, 2)
            target = batch[0][:, his_t:, 0]  # (batch_size, pred_t)
            output = model(src)  # (batch_size, his_t)
            loss = F.l1_loss(output[:, :min_len], target[:, :min_len])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss}')

        # Evaluation loop
        model.eval()
        total_test_loss = 0
        total_test_time = 0
        with torch.no_grad():
            for batch in test_loader:
                src = batch[0][:, :his_t, :]
                target = batch[0][:, his_t:, 0]
                start = time.time()
                output = model(src)
                end = time.time()
                loss = F.l1_loss(output[:, :min_len], target[:, :min_len])
                total_test_loss += loss.item()
                total_test_time += (end - start)
        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_time = total_test_time / len(test_loader)
        print(f'Test Loss: {avg_test_loss} | Test Time: {avg_test_time}s')

        # Step the scheduler
        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr}')

    model.save_model(num_epochs)
