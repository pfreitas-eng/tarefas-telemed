import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, tv_tensors
from torchvision.transforms import v2

source_data_dir = './source_data'
model_path = 'resnet18_chest_xray.pt'
report_path = '3_network_report.txt'
class_names = ['COVID-19', 'Lung-Opacity',
               'Normal', 'Viral Pneumonia', 'Tuberculosis']
epochs = 10
batch_size = 64
learning_rate = 1e-3
seed = 42
print_every = 100

torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


def load_split(split):
    # arquivos gerados pelo 1_preprocessamento.py (canal 0, [0, 1] e z-score do treino já aplicados)
    with np.load(os.path.join(source_data_dir, f'Dataset5_clean_{split}.npz')) as data:
        images = data['image']
        labels = data['image_label'].astype(np.int64).ravel()
    assert len(images) == len(labels)
    print(
        f"Carregado {split}: {len(labels)} imagens | shape {images.shape[1:]}")
    return images, labels


class ChestXrayDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # (H, W) -> (1, H, W) para o formato de canal esperado pelo torchvision
        image = tv_tensors.Image(torch.from_numpy(self.images[idx][None]))
        return self.transform(image), self.labels[idx]


def build_model():
    # ResNet-18 (arxiv 1512.03385) treinada do zero, com a primeira convolução aceitando 1 canal
    net = models.resnet18(weights=None, num_classes=len(class_names))
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                          padding=3, bias=False)
    return net.to(device)


print("\nCarregando dados...")
train_images, train_labels = load_split('train')
val_images, val_labels = load_split('val')
test_images, test_labels = load_split('test')

# a normalização já foi feita no pré-processamento, então aqui só há o flip de augmentation do treino
train_transform = v2.RandomHorizontalFlip(p=0.5)
eval_transform = v2.Identity()

# num_workers=0 evita copiar os arrays grandes para outros processos (Windows)
train_loader = DataLoader(ChestXrayDataset(train_images, train_labels, train_transform),
                          batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(ChestXrayDataset(val_images, val_labels, eval_transform),
                        batch_size=batch_size, num_workers=0)
test_loader = DataLoader(ChestXrayDataset(test_images, test_labels, eval_transform),
                         batch_size=batch_size, num_workers=0)

model = build_model()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
print(
    f"Modelo ResNet-18 criado | parâmetros: {total_params:,} (treináveis: {trainable_params:,})")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def run_epoch(loader, train):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % print_every == 0:
                    print(
                        f"  batch {batch_idx}/{len(loader)} | loss {loss.item():.4f}")
            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


history = []
best_val_loss = float('inf')
best_epoch = 0
best_state = None
start_time = time.time()
for epoch in range(epochs):
    epoch_start = time.time()
    print(f"\nÉpoca {epoch + 1}/{epochs} - treinando...")
    train_loss, train_acc = run_epoch(train_loader, train=True)
    print(f"Época {epoch + 1}/{epochs} - validando...")
    val_loss, val_acc = run_epoch(val_loader, train=False)
    epoch_time = time.time() - epoch_start
    history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc})
    print(f"Época {epoch + 1}/{epochs} | treino loss {train_loss:.4f} acc {train_acc:.4f} "
          f"| validação loss {val_loss:.4f} acc {val_acc:.4f} | tempo {epoch_time:.1f}s")
    # guarda os pesos da época com menor loss de validação
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())
        print(f"  novo melhor modelo (validação loss {val_loss:.4f})")

total_time = time.time() - start_time
print(
    f"\nTreinamento concluído em {total_time / 60:.1f} min | melhor época: {best_epoch}")

print("\nAvaliando no conjunto de teste...")
model.load_state_dict(best_state)
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        predictions.append(model(images.to(device)).argmax(1).cpu())
y_pred = torch.cat(predictions).numpy()
test_report = classification_report(
    test_labels, y_pred, target_names=class_names, digits=4)
test_confusion = confusion_matrix(test_labels, y_pred)

print("\nMétricas no conjunto de teste:")
print(test_report)

torch.save({'state_dict': best_state, 'class_names': class_names,
           'best_epoch': best_epoch}, model_path)
print("Modelo salvo em:", model_path)

lines = [
    "RELATÓRIO DA REDE - CLASSIFICAÇÃO DE IMAGENS (Dataset5)",
    f"Gerado em: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "",
    "PARÂMETROS DA REDE",
    "Arquitetura: ResNet-18 (treinada do zero, sem pesos pré-treinados)",
    "Entrada: 1 canal, 224x224, normalizada com média e desvio do treino",
    f"Saída: {len(class_names)} classes ({', '.join(class_names)})",
    f"Parâmetros totais: {total_params:,}",
    f"Parâmetros treináveis: {trainable_params:,}",
    "",
    "TREINAMENTO",
    f"Dispositivo: {device}",
    f"Épocas: {epochs}",
    f"Batch size: {batch_size}",
    f"Otimizador: Adam (lr={learning_rate})",
    "Função de perda: CrossEntropyLoss",
    "Augmentation: RandomHorizontalFlip(p=0.5), somente no treino",
    f"Seed: {seed}",
    f"Melhor época (menor loss de validação): {best_epoch}",
    f"Tempo total de treino: {total_time / 60:.1f} min",
    "",
    "HISTÓRICO DE TREINO",
    "Época | treino loss | treino acc | validação loss | validação acc",
]
for h in history:
    lines.append(f"{h['epoch']:>5} | {h['train_loss']:>11.4f} | {h['train_acc']:>10.4f} | "
                 f"{h['val_loss']:>14.4f} | {h['val_acc']:>14.4f}")
lines += [
    "",
    "MÉTRICAS NO CONJUNTO DE TESTE",
    test_report,
    "Matriz de confusão (linhas: classe real, colunas: classe predita):",
    np.array2string(test_confusion),
    "",
    "ARQUITETURA COMPLETA",
    str(model),
    "",
    f"Modelo salvo em: {model_path}",
    "Para carregar: net = build_model(); net.load_state_dict(torch.load(path, map_location=device)['state_dict'])",
]
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines) + "\n")
print("Relatório salvo em:", report_path)
