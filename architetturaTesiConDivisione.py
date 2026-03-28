import os
import cv2
import re
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights
from datetime import datetime

# Librerie di Augmentation
import albumentations as A
import torchvision.transforms as transforms
from pathlib import Path

# =====================================================================
# 1. PIPELINE DI DATA AUGMENTATION
# =====================================================================

train_geom_transform = A.Compose([
    A.Resize(height=280, width=280),

    A.OneOf([
        A.CenterCrop(height=256, width=256),
        A.RandomCrop(height=256, width=256),
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0))
    ], p=1.0),

    A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0, p=0.3),
    A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.2, p=0.5),

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    A.RandomRotate90(p=0.3),
    A.Rotate(limit=180, p=0.7)
])

pytorch_tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_geom_transform = A.Compose([
    A.Resize(height=256, width=256)
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
    def __init__(self, dataframe, images_dir, masks_dir, is_train=True):
        self.is_train = is_train
        self.df = dataframe

        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        img_map = {}
        if images_dir.exists():
            for p in images_dir.rglob("*.*"):
                if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    img_map[normalizza_nome(p.name)] = p

        mask_map = {}
        if masks_dir.exists():
            for p in masks_dir.rglob("*.*"):
                if p.is_file() and p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    mask_map[normalizza_nome(p.name)] = p

        self.data_pairs = []

        for idx, row in self.df.iterrows():
            chiave_csv = normalizza_nome(row['ID'])

            if chiave_csv in img_map and chiave_csv in mask_map:
                tumor_type_str = str(row['Pathology']).strip().lower()
                bm_label = 0.0 if "benign" in tumor_type_str else 1.0
                birads_label = int(row['BIRADS']) - 1

                self.data_pairs.append({
                    'id_name': chiave_csv,
                    'img_path': img_map[chiave_csv],
                    'mask_path': mask_map[chiave_csv],
                    'bm_label': bm_label,
                    'birads_label': birads_label
                })

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]

        image = cv2.imread(str(item['img_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_image = cv2.resize(image, (256, 256))

        mask = cv2.imread(str(item['mask_path']), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        if self.is_train:
            augmented = train_geom_transform(image=image, mask=mask)
        else:
            augmented = test_geom_transform(image=image, mask=mask)

        image = augmented['image']
        mask = augmented['mask']

        image_tensor = pytorch_tensor_transforms(image)

        mask_tensor = torch.from_numpy(mask).float()
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        bm_tensor = torch.tensor([item['bm_label']], dtype=torch.float32)
        birads_tensor = torch.tensor(item['birads_label'], dtype=torch.long)

        return image_tensor, mask_tensor, bm_tensor, birads_tensor, item['id_name'], orig_image


# =====================================================================
# 3. ARCHITETTURA DI RETE CON HOOKS PER GRAD-CAM
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

        self.cam_gradients = None
        self.cam_activations = None

    def save_gradients(self, grad):
        self.cam_gradients = grad

    def forward(self, x, extract_cam=False):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        bottleneck = self.encoder4(e3)

        if extract_cam:
            self.cam_activations = bottleneck
            bottleneck.register_hook(self.save_gradients)

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
# 4. TRAINING LOOP CON STAMPA IN TEMPO REALE E GENERAZIONE CAM
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
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    # Aggiunto enumerate per tenere traccia del numero del batch corrente
    for i, data in enumerate(dataloader):
        images, masks, bm_labels, birads_labels = data[0].to(device), data[1].to(device), data[2].to(device), data[
            3].to(device)

        optimizer.zero_grad()

        pred_masks_logits, pred_bm, pred_birads = model(images)

        loss_seg = criteria['seg_bce'](pred_masks_logits, masks) + criteria['seg_dice'](pred_masks_logits, masks)
        loss_bm = criteria['bm'](pred_bm, bm_labels)
        loss_birads = criteria['birads'](pred_birads, birads_labels)

        total_loss = loss_seg + loss_bm + loss_birads

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        # LA TUA MODIFICA: Stampa dinamica del progresso durante il Training
        # Usiamo '\r' per sovrascrivere la riga corrente invece di creare una nuova riga ogni volta
        print(f"   -> Progresso Batch: [{i + 1}/{total_batches}] | Loss Attuale: {total_loss.item():.4f}", end='\r',
              flush=True)

    # Andiamo a capo alla fine dell'epoca per non sovrascrivere la stampa finale
    print()
    return running_loss / total_batches


def generate_and_save_cam(model, image_tensor, orig_image, true_mask, img_id, save_folder, device):
    model.eval()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_()

    _, pred_bm, _ = model(image_tensor, extract_cam=True)
    pred_class = torch.sigmoid(pred_bm)

    model.zero_grad()
    pred_class.backward(retain_graph=True)

    gradients = model.cam_gradients[0].cpu().data.numpy()
    activations = model.cam_activations[0].cpu().data.numpy()
    weights = np.mean(gradients, axis=(1, 2))

    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig_image_bgr = cv2.cvtColor(orig_image.numpy(), cv2.COLOR_RGB2BGR)
    cam_overlay = cv2.addWeighted(orig_image_bgr, 0.5, heatmap, 0.5, 0)

    mask_bgr = true_mask.squeeze().numpy() * 255
    mask_bgr = mask_bgr.astype(np.uint8)
    mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_GRAY2BGR)

    combined_image = np.hstack((orig_image_bgr, cam_overlay, mask_bgr))

    cv2.putText(combined_image, 'Originale', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_image, 'Rete (CAM)', (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_image, 'Dottore (Mask)', (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    save_path = os.path.join(save_folder, f"{img_id}_cam.png")
    cv2.imwrite(save_path, combined_image)


def test_model(model, dataloader, device, save_cam_folder):
    model.eval()

    total_dice = 0.0
    bm_correct = 0
    birads_correct = 0
    total_samples = 0

    print(f"\nGenerazione e salvataggio delle CAM nella cartella: {save_cam_folder} ...")

    for data in dataloader:
        images, masks = data[0].to(device), data[1].to(device)
        bm_labels, birads_labels = data[2].to(device), data[3].to(device)
        id_names = data[4]
        orig_images = data[5]

        with torch.no_grad():
            pred_masks_logits, pred_bm, pred_birads = model(images)

            pred_masks_binary = (torch.sigmoid(pred_masks_logits) > 0.5).float()
            intersection = (pred_masks_binary * masks).sum()
            dice = (2. * intersection) / (pred_masks_binary.sum() + masks.sum() + 1e-6)
            total_dice += dice.item() * images.size(0)

            pred_bm_class = (torch.sigmoid(pred_bm) > 0.5).float()
            bm_correct += (pred_bm_class == bm_labels).sum().item()

            pred_birads_class = torch.argmax(pred_birads, dim=1)
            birads_correct += (pred_birads_class == birads_labels).sum().item()

            total_samples += images.size(0)

        for i in range(len(images)):
            if total_samples - len(images) + i < 15:
                img_tensor_cpu = images[i].cpu()
                generate_and_save_cam(
                    model, img_tensor_cpu, orig_images[i], masks[i].cpu(),
                    id_names[i], save_cam_folder, device
                )

    print("\n" + "=" * 40)
    print("      RISULTATI FASE DI TESTING      ")
    print("=" * 40)
    if total_samples > 0:
        print(f"-> Precisione Segmentazione (Dice): {total_dice / total_samples * 100:.2f}%")
        print(f"-> Accuratezza Benigno/Maligno:     {bm_correct / total_samples * 100:.2f}%")
        print(f"-> Accuratezza Classi BI-RADS:      {birads_correct / total_samples * 100:.2f}%")
    else:
        print("Nessun dato trovato per il Testing!")
    print("=" * 40 + "\n")


# =====================================================================
# 5. START! PUNTO D'INGRESSO DEL PROGRAMMA
# =====================================================================

if __name__ == "__main__":

    IMAGES_DIR = "/Users/pierfrancesco/Desktop/BUSBRA/Images"
    MASKS_DIR = "/Users/pierfrancesco/Desktop/BUSBRA/Masks"
    CSV_FILE = "/Users/pierfrancesco/Desktop/BUSBRA/bus_data.csv"
    FOLD_CSV = "/Users/pierfrancesco/Desktop/BUSBRA/5-fold-cv.csv"

    TEST_FOLD = 1
    NUM_EPOCHS = 3

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SAVE_PATH = f"/Users/pierfrancesco/Desktop/BUSBRA/modello_busbra_fold{TEST_FOLD}_{current_time}.pth"

    BASE_CAM_FOLDER = "/Users/pierfrancesco/Desktop/BUSBRA/CAM_Results"
    RUN_CAM_FOLDER = os.path.join(BASE_CAM_FOLDER, f"Fold_{TEST_FOLD}_{current_time}")
    os.makedirs(RUN_CAM_FOLDER, exist_ok=True)

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
        df_dati = pd.read_csv(CSV_FILE)
        df_folds = pd.read_csv(FOLD_CSV)

        fold_col = 'kFold'
        id_col = 'ID'

        df_folds[id_col] = df_folds[id_col].astype(str)
        test_ids = df_folds[df_folds[fold_col] == TEST_FOLD][id_col].tolist()

        df_dati['ID_str'] = df_dati['ID'].astype(str)
        df_train = df_dati[~df_dati['ID_str'].isin(test_ids)].copy()
        df_test = df_dati[df_dati['ID_str'].isin(test_ids)].copy()

        train_dataset = BUSBRA_CSV_Dataset(df_train, IMAGES_DIR, MASKS_DIR, is_train=True)
        test_dataset = BUSBRA_CSV_Dataset(df_test, IMAGES_DIR, MASKS_DIR, is_train=False)

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

        print(f"\n--- INIZIO ADDESTRAMENTO (FOLD {TEST_FOLD}) PER {NUM_EPOCHS} EPOCHE ---")

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoca [{epoch}/{NUM_EPOCHS}] avviata:")
            epoch_loss = train_one_epoch(model, train_loader, optimizer, criteria, device)
            print(f"-> Fine Epoca [{epoch}/{NUM_EPOCHS}] | Loss media calcolata: {epoch_loss:.4f}")

        print("\n--- ADDESTRAMENTO TERMINATO ---")
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Modello addestrato salvato in: {SAVE_PATH}")

        print(f"\nAvvio della fase di Testing sui dati del Fold {TEST_FOLD}...")
        test_model(model, test_loader, device, RUN_CAM_FOLDER)

    except Exception as e:
        print(f"\nERRORE IMPREVISTO: {e}")