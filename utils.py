import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# TORCH LOOPS
def train_student(model, train_loader, criterion=None, num_epochs=10, optimizer=None, device=None, save_path=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    model.train()

    for epoch in range(num_epochs):
        cum_loss = 0.0
        total = 0

        progress_bar = tqdm(train_loader, leave=True)
        progress_bar.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({'GT' if GROUND_TRUTH else 'WTSG'})")
        for idx, (images, true_labels, weak_labels) in enumerate(progress_bar):
            images, true_labels, weak_labels = images.to(device), true_labels.to(device), weak_labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if GROUND_TRUTH:
                # training step for ground truth model
                loss = criterion(outputs, true_labels)
            else:
                # training step for weak labels model
                loss = criterion(outputs, weak_labels)

            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            total += len(true_labels)

            avg_loss = cum_loss / total

            # Update the tqdm bar with loss and epoch
            progress_bar.set_postfix({"avg_loss":avg_loss, "cum_loss":cum_loss})
    
    if save_path is not None:
        torch.save(model.state_dict(), save_name)

def train_teacher(model, loader, criterion, optimizer=None, device=None):
    pass


def generate_pseudolabels(model, loader, criterion, device=None):
    pass


def test_loop(model, test_loader, criterion, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()

    # Initialize metrics
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient computation to save memory and computation
        progress_bar = tqdm(test_loader, leave=True)
        for idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device) # send data to gpu

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()  # Accumulate loss

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

            # Calculate metrics for display
            avg_loss = test_loss / total
            accuracy = 100 * correct / total

            # Update the tqdm bar with loss and accuracy
            progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

    # Calculate final metrics
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


# UTILS SPECIFIC TO WTSG
class WeakLabeledData(Dataset):
    '''A torch Dataset that returns both ground truth labels and weak pseudolabels'''
    def __init__(self, original_dataset: Dataset, weak_labels):
        self.dataset = original_dataset
        self.weak_labels = weak_labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, true_label = self.dataset[index]
        weak_label = self.weak_labels[index]
        return image, true_label, weak_label


def calculate_pgr(weak_accuracy, strong_ceiling, wts_accuracy):
    return (wts_accuracy - weak_accuracy) / (strong_ceiling - weak_accuracy)


# SETUP FOR MODELS USED IN MY EXPERIMENT
def setup_resnet50(num_classes=10, freeze=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if freeze:
        # freeze layers for finetuning
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze last layer to finetune
        for param in model.fc.parameters():
            param.requires_grad = True
        
        for param in model.layer4.parameters():
            param.requires_grad = True
            
    return model


class DinoClassification(nn.Module):
    """Add a classification head to a pretrained DINO model"""
    def __init__(self, original_model, num_classes=10):
        super(DinoClassification, self).__init__()
        
        # copy layers from original model
        self.dino = original_model
        
        # add classification head
        self.head = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # Pass input through dino model
        x = self.dino(x)

        # Extract the class token and pass through the classification head
        cls_token = x[:, 0]  # Shape: (batch_size, embed_dim)
        x = self.head(cls_token)  # Shape: (batch_size, num_classes)

        return x

    
def setup_dino(num_classes=10, freeze=True):
    # DINO, pretrained on ImageNet-1k based on the representations of ViT-L/16
    dino = torch.hub.load("BenediktAlkin/torchhub-ssl", "in1k_dinov2_l16", trust_repo=True)
    
    # add classification head
    model = DinoClassification(dino, num_classes)
    
    if freeze:
        # freeze layers for finetuning
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze last layer to finetune
        for param in model.head.parameters():
            param.requires_grad = True
        
        # for param in model.dino.blocks[-1].parameters():
        #     param.requires_grad = True
        
    return model


def setup_alexnet(num_classes=10, freeze=True):
    # Load the model
    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    
    alexnet.classifier[-1] = nn.Linear(alexnet.classifier[-1].in_features, num_classes)
    
    if freeze:
        # freeze layers for finetuning
        for param in alexnet.parameters():
            param.requires_grad = False

        # unfreeze last layer to finetune
        for param in alexnet.classifier.parameters():
            param.requires_grad = True
    
    return alexnet


def evaluate_pseudolabels(wtsg_loader):
    progress_bar = tqdm(wtsg_loader, leave=True)
    total = 0
    agree = 0
    for idx, (images, true_labels, weak_labels) in enumerate(progress_bar):
        agree += (true_labels == weak_labels).sum().item()
        total += len(true_labels)
        
        acc = 100 * agree / total
        
        progress_bar.set_postfix({"Weak Label Accuracy (%)": acc, "Correct": agree, "Total": total})
        
def count_unfrozen_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    