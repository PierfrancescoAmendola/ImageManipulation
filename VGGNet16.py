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
import random


# --- FUNZIONE PER LA RIPRODUCIBILITÀ ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed impostato a: {seed}")


# Funzione per mostrare immagini (utility)
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# --- BLOCCO PRINCIPALE ---
if __name__ == '__main__':

    print("--- CONFIGURAZIONE ---")

    # 1. INPUT UTENTE
    seed_val = int(input("Inserisci il seed (es. 42): "))
    fold = input("Inserisci il numero o il nome del fold o del test: ")
    epochNumber = int(input("Inserisci il numero di epoche: "))
    batch_size = int(input("Inserisci il numero di batch: "))
    lr = float(input("Inserisci il numero di learning rate: "))
    gpu_id = input("Inserisci il numero di gpu (es. 0, 1, 2...): ")

    # 2. SETUP BASE (Seed e Data)
    set_seed(seed_val)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 3. SETUP DEVICE (GPU/CPU)
    if torch.cuda.is_available() and gpu_id:
        device_str = f"cuda:{gpu_id}"  # Converte "2" in "cuda:2"
    else:
        device_str = "cpu"

    device = torch.device(device_str)
    print(f'Using device: {device}')

    # 4. PREPARAZIONE DATI
    # Definizione delle trasformazioni per VGG16 (richiede Resize a 224x224)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Generatore per il seed dei worker (sicurezza aggiuntiva)
    g = torch.Generator()
    g.manual_seed(seed_val)

    # Caricamento Training
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0, worker_init_fn=np.random.seed(seed_val))

    # Caricamento Test
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 5. SETUP TENSORBOARD (Aggiornato per VGG16)
    str_log = f'run_batch_{batch_size}_lr_{lr}_epoch{epochNumber}_Fold_{fold}_seed_{seed_val}_data_{date}'
    writer = SummaryWriter('runs/SuiteTest_1_VGG16/' + str_log,
                           comment=f"{str_log} batch_{batch_size} lr_{lr} epoch{epochNumber} Fold_{fold}")

    # 6. DEFINIZIONE MODELLO (VGG16)
    print("Caricamento modello VGG16...")
    net = torchvision.models.vgg16(weights='IMAGENET1K_V1')

    # Modifica dell'ultimo layer per CIFAR-10 (10 classi) per VGG16
    # L'ultimo strato di VGG16 è il sesto elemento (indice 6) nel blocco 'classifier'
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, 10)

    # Sposta la rete sul device (GPU o CPU)
    net = net.to(device)

    # 7. LOSS E OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # Visualizzazione iniziale (un batch)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # --- TRAINING LOOP ---
    print("Inizio Training...")
    for epoch in range(epochNumber):
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        net.train()  # Assicura che la rete sia in modalità training

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # Sposta i dati sulla GPU corretta
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            lossTraining = criterion(outputs, labels)
            lossTraining.backward()
            optimizer.step()

            running_loss += lossTraining.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracyTraining = 100 * correct_train / total_train

        print(f'[{epoch + 1}] Training loss: {avg_loss:.3f} - Training accuracy: {accuracyTraining:.2f}%')

        writer.add_scalar('training loss', avg_loss, epoch)
        writer.add_scalar('training accuracy', accuracyTraining, epoch)

        # Mostriamo grafo solo alla prima epoca
        if epoch == 0:
            writer.add_image('images', torchvision.utils.make_grid(images), 0)
            writer.add_graph(net, images.to(device))

        # --- TEST LOOP ---
        correct_test = 0
        total_test = 0
        running_lossTest = 0

        net.eval()  # Importante: mette la rete in modalità valutazione
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)

                lossTesting = criterion(outputs, labels)
                running_lossTest += lossTesting.item()

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        accuracyTest = 100 * correct_test / total_test
        avg_lossTest = running_lossTest / len(testloader)

        print(f'[{epoch + 1}] Test loss: {avg_lossTest:.3f} - Test accuracy: {accuracyTest:.2f}%')

        writer.add_scalar('test accuracy', accuracyTest, epoch)
        writer.add_scalar('test loss', avg_lossTest, epoch)

    print('Finished Training')

    # SALVATAGGIO MODELLO (Nome aggiornato per VGG16)
    PATH = f'./cifar_net_vgg16_fold{fold}_seed{seed_val}.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    writer.close()