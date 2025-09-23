import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model_code.resnet import resnet50
from model_code.trainer import CustomTrainer
from datetime import datetime

def run_finetuning(args):
    """
    Main function to run the fine-tuning process.
    """
    print(f"Starting fine-tuning process...")
    print(f"Mode: {args.mode}, Epochs: {args.epochs}, Learning Rate: {args.lr}")

    # --- 1. Setup Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 2. Setup DataLoaders ---
    if not os.path.isdir(args.data_dir):
         print(f"Error: Data directory not found at '{args.data_dir}'")
         return

    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Standard ImageFolder expects valid image files.
    # The user is instructed to provide them in the README.
    full_dataset = ImageFolder(root=args.data_dir, transform=data_transform)

    if args.mode == 'micro':
        # Use a small subset for quick testing
        num_samples = min(len(full_dataset), 8) # Use up to 8 samples for micro
        dataset = Subset(full_dataset, range(num_samples))
        print(f"Running in 'micro' mode with {len(dataset)} samples from {args.data_dir}.")
    else:
        dataset = full_dataset
        print(f"Running in 'full' mode with {len(dataset)} samples from {args.data_dir}.")

    if len(dataset) == 0:
        print("Error: No images found in the dataset directory. Please add images to the 'real' and 'fake' subfolders.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- 3. Load Pre-trained Model ---
    model = resnet50(num_classes=1)
    if not os.path.exists(args.weights_path):
        print(f"Error: Pre-trained weights not found at '{args.weights_path}'")
        print("Fine-tuning requires the pre-trained weights. Please download them first as instructed in README.md.")
        return

    state_dict = torch.load(args.weights_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(device)
    print("Pre-trained weights loaded successfully.")

    # --- 4. Initialize Trainer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = CustomTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device)

    # --- 5. Training Loop ---
    for epoch in range(args.epochs):
        epoch_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")

    # --- 6. Save the Fine-tuned Model ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_model_name = f"finetuned_model_{timestamp}.pth"
    output_model_path = os.path.join(args.output_dir, output_model_name)
    trainer.save_model(output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune the fake image detection model.")

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data directory (e.g., `training_data/micro`). Must contain "real" and "fake" subfolders.')
    parser.add_argument('--mode', type=str, default='micro', choices=['micro', 'full'], help="Training mode: 'micro' for a small test run, 'full' for the complete dataset.")
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--output_dir', type=str, default='finetuned_models', help='Directory to save the fine-tuned model.')
    parser.add_argument('--weights_path', type=str, default='weights/blur_jpg_prob0.5.pth', help='Path to the pre-trained weights to start fine-tuning from.')

    args = parser.parse_args()

    run_finetuning(args)
