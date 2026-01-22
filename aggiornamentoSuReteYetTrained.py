import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

transform = transforms.Compose([
    transforms.ToTensor(),
    #normalizziamo i dati in modo che abbiano media 0 e deviazione standard 1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size=1600

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
        #6 canali di input, 16 canali di output, kernel 5x5
        #filtro
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(9, 64, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
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
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


if __name__ == '__main__':
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Training
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                avg_loss = running_loss / 2000
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                writer.add_scalar('training loss', avg_loss, epoch * len(trainloader) + i)
                running_loss = 0.0

        # Mostriamo una griglia di immagini e il grafo solo alla prima epoca
        if epoch == 0:
            writer.add_image('images', torchvision.utils.make_grid(images), 0)
            writer.add_graph(net, images)

        # Calcolo dell'accuratezza (ripristinato)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                imgs, lbls = data
                outputs = net(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy after epoch {epoch + 1}: {accuracy:.2f}%')
        print(f'Loss after epoch {epoch + 1}: {loss:.2f}%')
        writer.add_scalar('accuracy', accuracy, epoch)

    print('Finished Training')
    print(f'La precisione del modello è: {accuracy:.2f}%')
    print(f'La loss del modello è: {loss.item():.4f}')
    
    # salviamo il modello
    PATH = './reteNuovaAddestrata/cifar_net_updated.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')

    # Aggiungiamo immagini e grafo al writer
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(net, images)

    # Mostriamo il grafico della precisione del modello con TensorBoardX (dati casuali)
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    writer.close()



#Ora ci occupiamo di stampare il grafico della precisione del modello con TensorBoardX

