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

## Uso

```bash
python prova.py
```

## Note

- Le immagini vengono lette dalla cartella `../images/` 
- I grafici vengono mostrati a schermo con `plt.show()`
- Le immagini elaborate vengono salvate come `.png` nella directory corrente
