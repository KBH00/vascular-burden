import argparse
from Data.dataload import get_dataloaders
from models.models import FeatureReconstructor
from utils.pytorch_ssim import SSIMLoss
import torch
import torch.optim as optim
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train Feature Autoencoder on 3D DICOM Images')
    parser.add_argument('--csv_path', type=str, default="C:/Users/kbh/Desktop/CNI/test/updated_subject_paths.csv", help='Path to the CSV file containing DICOM paths and labels')
    parser.add_argument('--train_base_dir', type=str, default="/home/kbh/Downloads/nii", help='Base directory for training DICOM files')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory to save model checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    return parser.parse_args()

def log_message(log_file, message):
    """Helper function to log messages to a text file and print to console."""
    with open(log_file, 'a') as f:
        f.write(message + '\n')
    print(message)

def train_model(args, batch_size, lr, image_size, dropout, hidden_dims, loss_fn, optimizer_choice, log_file):
    train_loader, validation_loader, test_loader = get_dataloaders(
        train_base_dir=args.train_base_dir,
        modality="FLAIR",  # Assuming you want to fix this, but could be parameterized too
        batch_size=batch_size,
        transform=None,  
        validation_split=0.1,
        seed=42
    )

    from argparse import Namespace
    config = Namespace()
    config.image_size = image_size
    config.hidden_dims = hidden_dims
    config.generator_hidden_dims = list(reversed(hidden_dims))
    config.discriminator_hidden_dims = hidden_dims
    config.dropout = dropout
    config.extractor_cnn_layers = ['layer1', 'layer2']
    config.keep_feature_prop = 1.0
    config.random_extractor = False
    config.loss_fn = loss_fn

    model = FeatureReconstructor(config).to(args.device)

    # Choose optimizer
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_val_loss = float('inf')  # Track best validation loss
    best_model_path = None

    log_message(log_file, f"Starting training with batch size {batch_size}, learning rate {lr}, image size {image_size}, dropout {dropout}, hidden_dims {hidden_dims}, loss_fn {loss_fn}, optimizer {optimizer_choice}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (volumes, _) in enumerate(train_loader):
            volumes = volumes.to(args.device)
            
            B, C, D, H, W = volumes.shape
            volumes_slices = volumes.view(B * D, C, H, W)

            loss_dict = model.loss(volumes_slices)
            loss = loss_dict['rec_loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                log_message(log_file, f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        log_message(log_file, f"Epoch [{epoch}/{args.epochs}], Average Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for volumes, _ in validation_loader:
                volumes = volumes.to(args.device)
                B, C, D, H, W = volumes.shape
                volumes_slices = volumes.view(B * D, C, H, W)

                loss_dict = model.loss(volumes_slices)
                loss = loss_dict['rec_loss']

                val_loss += loss.item()

        val_loss /= len(validation_loader)
        log_message(log_file, f"Validation Loss: {val_loss:.4f}")

        # Check for the best validation loss and save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.save_dir, f"best_model_batchsize_{batch_size}_lr_{lr}_imgsize_{image_size}_dropout_{dropout}_hiddendims_{hidden_dims}_optimizer_{optimizer_choice}.pth")
            model.save(config, f"best_model.pth", directory=args.save_dir)
            log_message(log_file, f"Best model updated: {best_model_path} with validation loss: {val_loss:.4f}")

        scheduler.step()

    return best_model_path, best_val_loss

def main():
    args = parse_args()

    batch_sizes = [8, 12, 16]  # List of batch sizes to experiment with
    learning_rates = [1e-3, 5e-4, 1e-4]  # List of learning rates to experiment with
    image_sizes = [64, 128, 256]  # Different image sizes to experiment with
    dropout_rates = [0.2, 0.25, 0.3]  # Dropout rates to experiment with
    hidden_dims_list = [[100, 150, 200, 300], [128, 256, 512], [64, 128, 256]]  # Different hidden layer configurations
    loss_functions = ['ssim', 'mse']  # Different loss functions (could add more like 'l1', etc.)
    optimizers = ['adam', 'sgd']  # Different optimizers to try

    best_overall_model = None
    best_overall_loss = float('inf')

    # Iterate over all combinations of hyperparameters
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for image_size in image_sizes:
                for dropout in dropout_rates:
                    for hidden_dims in hidden_dims_list:
                        for loss_fn in loss_functions:
                            for optimizer_choice in optimizers:
                                # Define a log file for this particular experiment
                                log_file = os.path.join(args.save_dir, f"training_log_batchsize_{batch_size}_lr_{lr}_imgsize_{image_size}_dropout_{dropout}_hiddendims_{hidden_dims}_lossfn_{loss_fn}_optimizer_{optimizer_choice}.txt")
                                log_message(log_file, f"Starting experiment with batch size {batch_size}, learning rate {lr}, image size {image_size}, dropout {dropout}, hidden_dims {hidden_dims}, loss_fn {loss_fn}, optimizer {optimizer_choice}\n")
                                
                                best_model_path, best_val_loss = train_model(args, batch_size, lr, image_size, dropout, hidden_dims, loss_fn, optimizer_choice, log_file)

                                # Update overall best model
                                if best_val_loss < best_overall_loss:
                                    best_overall_loss = best_val_loss
                                    best_overall_model = best_model_path
                                    log_message(log_file, f"New best model: {best_overall_model} with validation loss: {best_overall_loss:.4f}")

    # Log the best model and its corresponding validation loss
    final_log_file = os.path.join(args.save_dir, "best_overall_model_log.txt")
    log_message(final_log_file, f"Best overall model: {best_overall_model} with validation loss: {best_overall_loss:.4f}")

if __name__ == "__main__":
    main()
