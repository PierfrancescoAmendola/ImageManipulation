import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose([
    transforms.ToTensor(),
    #normalizziamo i dati in modo che abbiano media 0 e deviazione standard 1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#indica quante immagini prendiamo alla volta
batch_size=32


#da qui prendiamo le immagini da studiare per questo abbiamo true, qui creiamo la nostra AI
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

#qui prendiamo le immagini da testare per questo abbiamo false, sono immagini che la rete non ha mai visto
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Definiamo una rete neurale convoluzionale
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

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Passiamo da 6 a 32 filtri (molta più capacità)
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1) 
#         self.pool = nn.MaxPool2d(2, 2)
#         # Passiamo da 16 a 64 filtri
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
#         # Dopo due pooling (32x32 -> 16x16 -> 8x8), la dimensione è 8x8
#         self.fc1 = nn.Linear(64 * 8 * 8, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # trasforma in un unico vettore
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


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
            if i % 200 == 199:    # stampiamo ogni 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # salviamo il modello
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print(f'Model saved to {PATH}')




    # --- TEST DEL MODELLO ---
    print('\nAnalisi di alcune immagini di test...')
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Mostra immagini di test
    imshow(torchvision.utils.make_grid(images))
    print('Etichette reali: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Carichiamo il modello salvato per sicurezza
    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    # Chiediamo alla rete cosa vede
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predizioni AI:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    # Calcoliamo la precisione su tutto il dataset di test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nPrecisione della rete su 10.000 immagini di test: {100 * correct // total} %')