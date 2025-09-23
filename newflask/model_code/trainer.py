import torch
import os

class CustomTrainer:
    """
    A simplified trainer class to handle the training loop.
    It saves the model in a format compatible with the project's loading function.
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        """Runs a single epoch of training."""
        self.model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device).float().unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def save_model(self, output_path):
        """
        Saves the model in a dictionary format compatible with our loading function,
        which expects a 'model' key.
        """
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # The state dict is nested under the 'model' key to match the original
        # repository's format and our loading function in detector.py
        state_to_save = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state_to_save, output_path)
        print(f"Model saved to: {output_path}")
