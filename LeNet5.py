import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torchvision import models
import random  # Import necessario per il seed Python


# --- FUNZIONE PER LA RIPRODUCIBILITÀ ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # qual'ora usassimo multi-GPU

    # Garantisce che le operazioni convoluzionali siano deterministiche
    # Nota: questo potrebbe rallentare leggermente l'addestramento
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed impostato a: {seed}")


# Definizione delle trasformazioni
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

print("--- CONFIGURAZIONE ---")
seed_val = input("Inserisci il seed (es. 42): ")  # Nuovo input per il seed
seed_val = int(seed_val)

fold = input("Inserisci il numero o il nome del fold o del test: ")
epochNumber = int(input("Inserisci il numero di epoche: "))
batch_size = int(input("Inserisci il numero di batch: "))
lr = float(input("Inserisci il numero di learning rate: "))
gpu_id = input("Inserisci il numero di gpu (es. 0, 1, 2...): ")

# --- APPLICAZIONE SEED ---
# È fondamentale chiamarlo PRIMA di creare dataloader o modelli
set_seed(seed_val)

# Costruzione stringa log
str_log = f'run_batch_{batch_size}_lr_{lr}_epoch{epochNumber}_Fold_{fold}_seed_{seed_val}_data_{date}'

writer = SummaryWriter('runs/SuiteTest_2_LeNet5/' + str_log,
                       comment=f"{str_log} batch_{batch_size} lr_{lr} epoch{epochNumber} Fold_{fold}")

if torch.cuda.is_available() and gpu_id:
    device_str = f"cuda:{gpu_id}"  # Converte "2" in "cuda:2"
else:
    device_str = "cpu"
# Setup Device
device = torch.device(device_str)
print(f'Using device: {device}')

# --- GENERATORE DATALOADER ---
# Per massima sicurezza, passiamo un generatore al DataLoader (utile se num_workers > 0)
g = torch.Generator()
g.manual_seed(seed_val)

# Caricamento dati Training
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Nota: worker_init_fn è necessario solo se num_workers > 0, ma lo lasciamo per completezza
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0, worker_init_fn=np.random.seed(seed_val))

# Caricamento dati Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Funzione per mostrare immagini
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Definizione della Rete Neurale
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 canali input (RGB), 6 output, kernel 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 6 canali input, 16 output, kernel 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Calcolo dimensioni:
        # Input: 32x32 -> Conv1: 28x28 -> Pool: 14x14
        # Conv2: 10x10 -> Pool: 5x5 -> Canali: 16
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

if __name__ == '__main__':

    # Prendiamo un batch di immagini casuali per visualizzazione iniziale
    # Grazie al seed, queste immagini saranno sempre le stesse per lo stesso seed
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Visualizzazione classi nel terminale
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

    # --- Training Loop ---
    for epoch in range(epochNumber):
        running_loss = 0.0
        total = 0
        correct = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            lossTraining = criterion(outputs, labels)
            lossTraining.backward()
            optimizer.step()

            running_loss += lossTraining.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracyTraining = 100 * correct / total

        print(f'[{epoch + 1}] Training loss: {avg_loss:.3f} - Training accuracy:{accuracyTraining:.2f}%')

        writer.add_scalar('training loss', avg_loss, epoch)
        writer.add_scalar('training accuracy', accuracyTraining, epoch)

        # Mostriamo grafo solo alla prima epoca
        if epoch == 0:
            writer.add_image('images', torchvision.utils.make_grid(images), 0)
            writer.add_graph(net, images.to(device))

        # --- Test Loop ---
        correct = 0
        total = 0
        running_lossTest = 0

        net.eval()  # Importante: mette la rete in modalità valutazione (disabilita dropout/batchnorm se presenti)
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)

                lossTesting = criterion(outputs, labels)
                running_lossTest += lossTesting.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        net.train()  # Rimettiamo in modalità training

        accuracyTest = 100 * correct / total
        avg_lossTest = running_lossTest / len(testloader)

        print(f'[{epoch + 1}] Test loss: {avg_lossTest:.3f} - Test accuracy:{accuracyTest:.2f}%')

        writer.add_scalar('test accuracy', accuracyTest, epoch)
        writer.add_scalar('test loss', avg_lossTest, epoch)

    print('Finished Training')

    PATH = f'./cifar_net_fold{fold}_seed{seed_val}.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    writer.close()