import os
import cv2
import re
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet34, ResNet34_Weights

# Librerie di Augmentation (Albumentations + PyTorch Nativo)
import albumentations as A
import torchvision.transforms as transforms
from pathlib import Path

# =====================================================================
# 1. PIPELINE DI DATA AUGMENTATION (Esattamente come l'hai chiesta)
# =====================================================================

# PARTE 1: Trasformazioni spaziali con Albumentations (sincronizza mask e image)
train_geom_transform = A.Compose([
    A.Resize(280, 280),  # Resize classico
    A.CenterCrop(256, 256, p=0.3),  # Crop classico
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=0.5),  # Random resize & Random crop

    A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0, p=0.3),  # Contrast fisso
    A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.2, p=0.5),  # Random contrast

    A.HorizontalFlip(p=0.5),  # Flip (destra-sinistra)
    A.VerticalFlip(p=0.5),  # Reverse / Turn down / Turn up (alto-basso)

    A.RandomRotate90(p=0.3),  # Turn 90 degree

    # Turn 45, 50, 180, 360 degree:
    # Il limite (-180, 180) copre matematicamente l'intero angolo giro.
    # 360 gradi equivale a 0 (nessuna rotazione), ed è coperto.
    A.Rotate(limit=180, p=0.7)
])

# PARTE 2: Utilizzo delle funzioni native di PyTorch (torchvision) per i Tensori
pytorch_tensor_transforms = transforms.Compose([
    transforms.ToTensor(),  # Funzione di PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Funzione di PyTorch
])

# Pipeline leggera per il Testing (nessuna rotazione, solo PyTorch tensorization)
test_geom_transform = A.Compose([
    A.Resize(256, 256)
])


# =====================================================================
# 2. DATALOADER: LETTURA E NORMALIZZAZIONE
# =====================================================================

def normalizza_nome(nome):
    n = str(nome).lower()
    n = re.sub(r'\.(png|jpg|jpeg|bmp)$', '', n)
    n = re.sub(r'^(bus_|masks_|mask_)', '', n)
    n = re.sub(r'(_mask|mask)$', '', n)
    return n.strip()


class BUSBRA_CSV_Dataset(Dataset):
    def __init__(self, csv_file, images_dir, masks_dir, is_train=True):
        self.is_train = is_train
        df_completo = pd.read_csv(csv_file)

        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        # Scansione Immagini
        img_map = {}
        if images_dir.exists():
            for p in images_dir.rglob("*.*"):
                if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    img_map[normalizza_nome(p.name)] = p

        # Scansione Maschere
        mask_map = {}
        if masks_dir.exists():
            for p in masks_dir.rglob("*.*"):
                if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    mask_map[normalizza_nome(p.name)] = p

        self.data_pairs = []

        # Accoppiamento
        for idx, row in df_completo.iterrows():
            chiave_csv = normalizza_nome(row['ID'])

            if chiave_csv in img_map and chiave_csv in mask_map:
                tumor_type_str = str(row['Pathology']).strip().lower()
                bm_label = 0.0 if "benign" in tumor_type_str else 1.0
                birads_label = int(row['BIRADS']) - 1

                self.data_pairs.append({
                    'img_path': img_map[chiave_csv],
                    'mask_path': mask_map[chiave_csv],
                    'bm_label': bm_label,
                    'birads_label': birads_label
                })

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]

        # Lettura
        image = cv2.imread(str(item['img_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(item['mask_path']), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # 1. Applicazione Augmentation geometriche (Albumentations)
        if self.is_train:
            augmented = train_geom_transform(image=image, mask=mask)
        else:
            augmented = test_geom_transform(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        # 2. Applicazione PyTorch Transforms
        image = pytorch_tensor_transforms(image)  # Converte in Tensore PyTorch e normalizza

        # Gestione Maschera in Tensore
        mask = torch.from_numpy(mask).float()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        bm_tensor = torch.tensor([item['bm_label']], dtype=torch.float32)
        birads_tensor = torch.tensor(item['birads_label'], dtype=torch.long)

        return image, mask, bm_tensor, birads_tensor


# =====================================================================
# 3. ARCHITETTURA DI RETE: MULTI-TASK U-NET + RESNET
# =====================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiTaskBreastCancerNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskBreastCancerNet, self).__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)
        self.segmentation_head = nn.Conv2d(64, 1, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_bm = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1)
        )
        self.classifier_birads = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 5)
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        bottleneck = self.encoder4(e3)

        d4 = self.decoder4(torch.cat((self.upconv4(bottleneck), e3), dim=1))
        d3 = self.decoder3(torch.cat((self.upconv3(d4), e2), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e1), dim=1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), e0), dim=1))

        mask_logits = self.segmentation_head(d1)
        mask_logits = F.interpolate(mask_logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        mask_probs = torch.sigmoid(mask_logits)

        attention_map = F.interpolate(mask_probs, size=bottleneck.shape[2:], mode='bilinear', align_corners=False)
        attended_features = bottleneck * attention_map

        pooled = torch.flatten(self.global_pool(attended_features), 1)

        return mask_logits, self.classifier_bm(pooled), self.classifier_birads(pooled)


# =====================================================================
# 4. TRAINING LOOP & TESTING LOOP
# =====================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        return 1 - ((2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth))


def train_one_epoch(model, dataloader, optimizer, criteria, device):
    model.train()  # Setta la rete in modalità addestramento
    running_loss = 0.0

    for images, masks, bm_labels, birads_labels in dataloader:
        images, masks = images.to(device), masks.to(device)
        bm_labels, birads_labels = bm_labels.to(device), birads_labels.to(device)

        optimizer.zero_grad()

        pred_masks_logits, pred_bm, pred_birads = model(images)

        loss_seg = criteria['seg_bce'](pred_masks_logits, masks) + criteria['seg_dice'](pred_masks_logits, masks)
        loss_bm = criteria['bm'](pred_bm, bm_labels)
        loss_birads = criteria['birads'](pred_birads, birads_labels)

        total_loss = loss_seg + loss_bm + loss_birads

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    return running_loss / len(dataloader)


def test_model(model, dataloader, device):
    """ Funzione di Testing della Rete """
    model.eval()  # Setta la rete in modalità test (congela i pesi)

    total_dice = 0.0
    bm_correct = 0
    birads_correct = 0
    total_samples = 0

    with torch.no_grad():  # Nessun gradiente = risparmio memoria e velocità
        for images, masks, bm_labels, birads_labels in dataloader:
            images, masks = images.to(device), masks.to(device)
            bm_labels, birads_labels = bm_labels.to(device), birads_labels.to(device)

            # Predizioni della rete
            pred_masks_logits, pred_bm, pred_birads = model(images)

            # 1. Calcolo Accuratezza Segmentazione (Dice Score Reale)
            pred_masks_binary = (torch.sigmoid(pred_masks_logits) > 0.5).float()
            intersection = (pred_masks_binary * masks).sum()
            dice = (2. * intersection) / (pred_masks_binary.sum() + masks.sum() + 1e-6)
            total_dice += dice.item() * images.size(0)

            # 2. Calcolo Accuratezza Benigno/Maligno
            pred_bm_class = (torch.sigmoid(pred_bm) > 0.5).float()
            bm_correct += (pred_bm_class == bm_labels).sum().item()

            # 3. Calcolo Accuratezza BI-RADS
            pred_birads_class = torch.argmax(pred_birads, dim=1)
            birads_correct += (pred_birads_class == birads_labels).sum().item()

            total_samples += images.size(0)

    # Stampa i risultati finali
    print("\n" + "=" * 40)
    print("      RISULTATI FASE DI TESTING      ")
    print("=" * 40)
    print(f"-> Precisione Segmentazione (Dice): {total_dice / total_samples * 100:.2f}%")
    print(f"-> Accuratezza Benigno/Maligno:     {bm_correct / total_samples * 100:.2f}%")
    print(f"-> Accuratezza Classi BI-RADS:      {birads_correct / total_samples * 100:.2f}%")
    print("=" * 40 + "\n")


# =====================================================================
# 5. START! PUNTO D'INGRESSO DEL PROGRAMMA
# =====================================================================

if __name__ == "__main__":

    IMAGES_DIR = "/Users/pierfrancesco/Desktop/BUSBRA/Images"
    MASKS_DIR = "/Users/pierfrancesco/Desktop/BUSBRA/Masks"
    CSV_FILE = "/Users/pierfrancesco/Desktop/BUSBRA/bus_data.csv"

    # PARAMETRI
    NUM_EPOCHS = 50
    SAVE_PATH = "/Users/pierfrancesco/Desktop/BUSBRA/modello_busbra_finale.pth"

    # OTTIMIZZAZIONE MAC
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        use_pin_memory = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        use_pin_memory = True
    else:
        device = torch.device("cpu")
        use_pin_memory = False

    print(f"Inizializzazione... Utilizzo del device: {device}")

    try:
        # Carica TUTTO il dataset
        full_dataset = BUSBRA_CSV_Dataset(CSV_FILE, IMAGES_DIR, MASKS_DIR, is_train=True)
        print(f"Totale file trovati e accoppiati: {len(full_dataset)}")

        # SPLIT DI PYTORCH: 80% Addestramento, 20% Testing
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        # Disattiviamo l'augmentation estrema per il Test set (vogliamo valutare i dati originali)
        test_dataset.dataset.is_train = False

        print(f"Immagini per Addestramento: {train_size} | Immagini per Testing: {test_size}")

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

        model = MultiTaskBreastCancerNet(pretrained=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        criteria = {
            'seg_bce': nn.BCEWithLogitsLoss(),
            'seg_dice': DiceLoss(),
            'bm': nn.BCEWithLogitsLoss(),
            'birads': nn.CrossEntropyLoss()
        }

        print(f"\n--- INIZIO ADDESTRAMENTO PER {NUM_EPOCHS} EPOCHE ---")

        for epoch in range(1, NUM_EPOCHS + 1):
            epoch_loss = train_one_epoch(model, train_loader, optimizer, criteria, device)
            print(f"Epoca [{epoch}/{NUM_EPOCHS}] | Loss media: {epoch_loss:.4f}")

        print("\n--- ADDESTRAMENTO TERMINATO ---")
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Modello addestrato salvato in: {SAVE_PATH}")

        # AVVIA LA FASE DI TESTING SUI DATI MAI VISTI!
        print("\nAvvio della fase di Testing sui dati di convalida...")
        test_model(model, test_loader, device)

    except Exception as e:
        print(f"\nERRORE IMPREVISTO: {e}")