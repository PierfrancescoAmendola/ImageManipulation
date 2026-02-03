# Image Manipulation

Script Python per manipolazione e analisi di immagini usando OpenCV, Matplotlib e Pandas.

## Funzionalità

- Lettura e visualizzazione di immagini (matplotlib e cv2)
- Analisi dei canali RGB
- Conversione di formati colore (BGR ↔ RGB, RGB → Grayscale)
- Ridimensionamento e scaling immagini
- Blur e filtri
- Salvataggio immagini elaborate

## Requisiti

```bash
pip install pandas numpy opencv-python matplotlib
```

## Setup Ambiente Virtuale

Per creare e attivare un ambiente virtuale:

```bash
python3 -m venv venv
source venv/bin/activate  # Su macOS/Linux
# Su Windows: venv\Scripts\activate
```

Installa le dipendenze per il training della rete neurale:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorboard matplotlib numpy
```

## Uso

Per eseguire gli script di manipolazione immagini:

```bash
python3 prova.py
```

Per addestrare la rete neurale su CIFAR-10:

```bash
python3 ImagePython/training.py
```

Per aggiornare una rete già addestrata:

```bash
python3 aggiornamentoSuReteYetTrained.py
```

## Visualizzazione Risultati con TensorBoard

Per vedere i log di training e i risultati:

```bash
tensorboard --logdir runs
```

Apri il browser e vai su `http://localhost:6006` o su `http://localhost:6007`

## Note

## Note

- Le immagini vengono lette dalla cartella `../images/` 
- I grafici vengono mostrati a schermo con `plt.show()`
- Le immagini elaborate vengono salvate come `.png` nella directory corrente
