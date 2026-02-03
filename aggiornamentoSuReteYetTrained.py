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

transform = transforms.Compose([
    transforms.ToTensor(),
    #normalizziamo i dati in modo che abbiano media 0 e deviazione standard 1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

lr = 0.0001
date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

fold = input("Inserisci il numero o il nome del fold o del test: ")
epochNumber = int(input("Inserisci il numero di epoche: "))
batch_size = int(input("Inserisci il numero di batch: "))

str_log = f'run_batch_{batch_size}_lr_{lr}_epoch{epochNumber}_Fold_{fold}_data_{date}'

writer = SummaryWriter('runs/' + str_log, comment=str_log + f" batch_{batch_size} lr_{lr} epoch{epochNumber} Fold_{fold}")


#per il training prendo i dati 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# prendiamo anche il testset per calcolare l'accuratezza
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#mostriamo l'immagine
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#ora definiamo la rete convulazionale che ci permette di effettuare l'addestramento 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #3 canali di input (RGB), 6 canali di output, kernel 5x5
        #filtro
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        #6 canali di input, 16 canali di output, kernel 5x5, è un filtro
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(9, 64, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #---------1. Il blocco visivo (Estrazione caratteristiche)---------
        #F.relu(...) viene applicata la funzione di attivazione ReLu. E' un filtro matematico che trasf. i num neg in zero.
        #Così facendo si turnOn solo i neuroni che hanno trovato qualcosa di interessante e turnOff quelli incerti
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        #---------2. Il ponte (Appiattimento)---------
        #torch.flatten: Prende tutti i pixel delle feature maps rimaste e li mette in fila indiana.
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        #---------3. Il blocco decisionale (Classificazione)---------
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Caricamento dei pesi della rete esistente
variabile=input("Vuoi riaddestrare la rete o non hai una rete addestrata? Si per riaddestrare no per addestrare da zero (s/n): ")
if variabile=="s":
    PATH_PREVIOUS =  './ImagePython/reteNuovaAddestrata/cifar_net_updated.pth'
    try:
        net.load_state_dict(torch.load(PATH_PREVIOUS, weights_only=True))
        print(f"Modello caricato correttamente da {PATH_PREVIOUS}")
    except FileNotFoundError:
        print(f"File {PATH_PREVIOUS} non trovato. Inizio addestramento da zero.")
else:
    PATH_PREVIOUS = './ImagePython/cifar_net.pth'
    try:
        net.load_state_dict(torch.load(PATH_PREVIOUS, weights_only=True))
        print(f"Modello caricato correttamente da {PATH_PREVIOUS}")
    except FileNotFoundError:
        print(f"File {PATH_PREVIOUS} non trovato. Inizio addestramento da zero.")

criterion = nn.CrossEntropyLoss()
#dobbiamo iniziare a giocare anche sul learningRate
#0.001 accuratezza salita esponenzialmente con 4 di batch_size
#0.00001 accuratezza scesa esponenzialmente con  4 di batch_size ogni volta diminuiva sempre più
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


if __name__ == '__main__':
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Training
    for epoch in range(epochNumber):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            lossTraining = criterion(outputs, labels)
            lossTraining.backward()
            optimizer.step()

            running_loss += lossTraining.item()

            # --- CALCOLO ACCURACY TRAINING ---
            # Ottieniamo la classe con la probabilità più alta
            _, predicted = torch.max(outputs.data, 1)
            # Aggiorniamo il numero totale di campioni
            total += labels.size(0)
            # Aggiorniamo il numero di previsioni corrette
            correct += (predicted == labels).sum().item()

        running_loss=running_loss / len(trainloader)
        accuracyTraining= 100 * correct / total

        print(f'[{epoch + 1}, {i:5d}] Training loss: {running_loss:.3f} - Training accuracy:{accuracyTraining:.2f}%')

        writer.add_scalar('training loss', running_loss, epoch)
        writer.add_scalar('training accuracy', accuracyTraining , epoch)

                

        # Mostriamo una griglia di immagini e il grafo solo alla prima epoca
        if epoch == 0:
            writer.add_image('images', torchvision.utils.make_grid(images), 0)
            writer.add_graph(net, images)

        # Calcolo dell'accuratezzaTest e della loss dei test
        correct = 0
        total = 0
        running_lossTest = 0

        with torch.no_grad():

            for data in testloader:
                inputs, labels = data
                # Forward pass
                outputs = net(inputs)
        
                # 1. Calcolo della Loss di Test
                lossTesting = criterion(outputs, labels)
                running_lossTest += lossTesting.item()
                
                # 2. Calcolo dell'Accuracy di Test
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracyTest = 100 * correct / total
        running_lossTest = running_lossTest / len(testloader)
        
        print(f'[{epoch + 1}, {i:5d}] Test loss: {running_lossTest:.3f} - Test accuracy:{accuracyTest:.2f}%')

        writer.add_scalar('test accuracy', accuracyTest, epoch)
        writer.add_scalar('test loss' , running_lossTest, epoch)

    print('Finished Training')

    # salviamo il modello
    PATH = './reteNuovaAddestrata/cifar_net_updated.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    # Aggiungiamo immagini e grafo al writer
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(net, images)
    
    writer.close()


