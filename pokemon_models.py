import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import time
from tqdm import tqdm
from config import *

# Dataset personalizado para Pokémon
class PokemonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [i for i in sorted(os.listdir(root_dir)) if i != '.DS_Store']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Modelo CNN
class PokemonCNN(nn.Module):
    def __init__(self, num_classes):
        super(PokemonCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Primera capa convolucional
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Segunda capa convolucional
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.feature_size = 128 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Modelo ResNet con transfer learning
class PokemonResNet(nn.Module):
    def __init__(self, num_classes):
        super(PokemonResNet, self).__init__()
        
        # Cargar ResNet50 pre-entrenado
        self.model = models.resnet50(pretrained=True)
        
        # Congelar los parámetros de las capas convolucionales
        for param in list(self.model.parameters())[:-20]:  # Dejamos las últimas capas sin congelar
            param.requires_grad = False
        
        # Reemplazar la capa fully connected final
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Funciones de entrenamiento y validación compartidas
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'batch': f'{batch_idx + 1}/{len(train_loader)}',
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'batch': f'{batch_idx + 1}/{len(val_loader)}',
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(val_loader), 100. * correct / total

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name, patience=5, min_delta=0.001):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f'Iniciando entrenamiento {model_name}...')
    print(f'Tamaño del batch: {BATCH_SIZE}')
    print(f'Número de batches por época: {len(train_loader)}')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f'\nNuevo mejor modelo guardado! (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - start_time
        print(f'\nÉpoca {epoch+1}/{num_epochs} ({epoch_time:.2f}s):')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if patience_counter >= patience:
            print(f'\nEarly stopping después de {epoch+1} épocas')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def predict_image(model, image_path, transform, device, idx_to_class):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return idx_to_class[predicted_class], confidence

def train_cnn():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Usando dispositivo: {device}')
    
    dataset = PokemonDataset(root_dir='data', transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_classes = len(dataset.classes)
    model = PokemonCNN(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        model_name='CNN'
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'idx_to_class': dataset.idx_to_class,
        'transform': transform
    }, 'pokemon_cnn_model.pth')
    print('Modelo guardado como pokemon_cnn_model.pth')

def train_resnet():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Usando dispositivo: {device}')
    
    dataset = PokemonDataset(root_dir='data', transform=resnet_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_classes = len(dataset.classes)
    model = PokemonResNet(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        model_name='ResNet'
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'idx_to_class': dataset.idx_to_class,
        'transform': resnet_transform
    }, 'pokemon_resnet_model.pth')
    print('Modelo guardado como pokemon_resnet_model.pth')

def predict_cnn(model_path, image_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PokemonCNN(len(checkpoint['idx_to_class']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    predicted_class, confidence = predict_image(
        model, 
        image_path, 
        checkpoint['transform'], 
        'cpu',
        checkpoint['idx_to_class']
    )
    
    print(f'Predicción: {predicted_class}')
    print(f'Confianza: {confidence:.2%}')

def predict_resnet(model_path, image_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = PokemonResNet(len(checkpoint['idx_to_class']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    predicted_class, confidence = predict_image(
        model, 
        image_path, 
        checkpoint['transform'], 
        'cpu',
        checkpoint['idx_to_class']
    )
    
    print(f'Predicción: {predicted_class}')
    print(f'Confianza: {confidence:.2%}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Entrenar o predecir con modelos de Pokémon')
    parser.add_argument('--mode', type=str, choices=['train_cnn', 'train_resnet', 'predict_cnn', 'predict_resnet'],
                      help='Modo de operación: entrenar CNN, entrenar ResNet, predecir con CNN o predecir con ResNet')
    parser.add_argument('--model_path', type=str, help='Ruta al modelo para predicción')
    parser.add_argument('--image_path', type=str, help='Ruta a la imagen para predicción')
    
    args = parser.parse_args()
    
    if args.mode == 'train_cnn':
        train_cnn()
    elif args.mode == 'train_resnet':
        train_resnet()
    elif args.mode == 'predict_cnn':
        if not args.model_path or not args.image_path:
            print('Se requieren --model_path y --image_path para predicción')
        else:
            predict_cnn(args.model_path, args.image_path)
    elif args.mode == 'predict_resnet':
        if not args.model_path or not args.image_path:
            print('Se requieren --model_path y --image_path para predicción')
        else:
            predict_resnet(args.model_path, args.image_path) 