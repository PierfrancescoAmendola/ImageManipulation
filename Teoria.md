
### self.conv1 = nn.Conv2d(3, 6, 5)

Definizione di rete convulazionale: Una Rete Neurale Convoluzionale (in inglese CNN, Convolutional Neural Network) è un tipo di intelligenza artificiale progettata specificamente per analizzare dati visivi, come foto o video. Mentre una rete neurale classica tratta i dati come una lunga lista di numeri senza ordine, la CNN è progettata per capire la geometria dell'immagine (cosa sta vicino a cosa).

Abbiamo 3 fasi prinicipali:
1) Fase di **Estrazione** (convulazione + ReLu): I primi strati riconoscono linee semplici, curve e colori (bordi). Gli strati successivi mettono insieme queste linee per riconoscere forme più complesse (occhi, orecchie, ruote).

2) Fase di **Riduzione** (pooling): Dopo aver trovato le caratteristiche (es. "qui c'è una curva"), la rete riduce la dimensione dell'immagine per alleggerire i calcoli. Concentrarsi solo sulle cose importanti, ignorando i dettagli inutili (es. non importa la posizione esatta del pixel dell'orecchio, basta sapere che l'orecchio c'è).

3) Fase di **Classificazione** (fully connected): Dopo vari passaggi di Convoluzione e Pooling, l'immagine non è più una foto, ma una lista di numeri astratti che descrivono il contenuto (es. "ha i baffi", "ha le orecchie a punta", "è peloso").
Questi dati vengono passati a una rete neurale classica (chiamata spesso Dense o Linear in PyTorch) che dà il verdetto finale. Output: "È un Gatto al 98%".


Conv2d sta per Convoluzione a 2 Dimensioni. Quest'ultima viene considerata come l'operazione matematica fondamentale per la visione artificiale.

Il processo è composto da 4 fasi:
1) **Scorrimento**: il kernel si posizione su una porzione dell'immagine
2) **Moltiplicazione**: i valori dei pixel dell'immagine vengono moltiplicati per i valori (pesi), che sono contenuti all'interno del kernel
3) **Somma**: tutti i risultati poi vengono sommati affinchè si possa ottenere un unico numero
4) **Output**: questo numero, ottenuto mediante la somma, diventerà un pixel nella nuova immagine, chiamata feauture map

usiamo tutto ciò perchè a differenze delle reti neurali classiche, che trattano ogni pixel come un dato isolato, la conv2D preserva la relazione spaziale tra i pixel. Serve ad estrarre caratteristiche visive come i bordi, le linee, angoli o texture.

Con out_channels=6, la rete userà 6 kernel diversi contemporaneamente, ognuno specializzato nel trovare qualcosa di diverso

---

### self.pool = nn.MaxPool2d(2, 2)

Il layer MaxPool viene inserito subito dopo uno convulazionale, ciò consente di ridurre le dimensioni dei dati

La funzione accetta (kernel_size, stride).
1) Il primo 2 (kernel_size): È la dimensione della finestra. La rete prenderà in considerazione un quadratino di 2x2 pixel.
2) Il secondo 2 (stride): È il "passo" o l'andatura. Significa che dopo aver analizzato un quadratino, la finestra si sposta di 2 pixel alla volta (non si sovrappone).

Il Max pooling è un operazione di semplificazione che punta all'aggressività. Il suo scopo è quello di entrare all'interno della finestra 2x2 e tenere solo il numero più alto all'interno eliminando i restanti all'interno. Così facendo otteniamo soltanto il massimo, e non ci preouccupiamo di tenere anche gli scarti

1 1 | 2 4 \
5 6 | 7 8 \
3 2 | 1 0 \
1 2 | 3 4

se applichiamo il Pool otteremo che il primo blocco avrà i numeri {1,1,5,6} ed il massimo nell'insieme è 6. Quindi manterremo il 6. Nel secondo è 8, terzo 3, quarto 4

così facendo ridurriamo la matrice, o meglio la feature map ottenendo così solamente 4 numeri, che sono i valori più importanti poichè sono i massimi

6 | 8\
3 | 4

Così facendo l'immagine satà dimezzata e ciò faciliterà la rete nell'assimilarla ma sopratutto sarà più veloce da calcolare

Un altro motivo rilevante è la capacità di mantenere informazioni rilevanti. se la convoluzione precedente ha trovato un "bordo" o una "forma" importante (valore alto), il Max Pooling lo conserva e scarta il "rumore" (valori bassi) intorno ad esso. Così facendo otterremmo un immagine più pulita, chiara, e senza rumore, che inserita all'interno della rete è più facile da assimilare ma sopratutto da poter calcolare.

### self.fc1 = nn.Linear(16 * 5 * 5, 120)

**fc** sta per fully connected. nelle reti neurali classiche tutti i layer sono fully connected. invece nelle reti neurali CNN, cioè quelle convulazionali si utilizzano questi strati, fully connected, solo alla fine per prendere la decisione finale

**nn.Linear** è il comando PyTorch per creare uno strato denso (o fully connected).
Mentre Conv2d connette i neuroni solo a quelli vicini (finestra locale), Linear connette tutti i neuroni di ingresso con tutti i neuroni di uscita.
Analizziamo self.fc1 = nn.Linear(16 * 5 * 5, 120):

**L'Input (16 * 5 * 5):**
Qui avviene un passaggio cruciale chiamato Flattening (appiattimento).
L'output della conv2 (dopo il pooling) è un volume 3D: immagina 16 piccole immagini (canali), ognuna grande 5x5 pixel.
Per poter passare questo volume a uno strato Linear, dobbiamo "srotolarlo" in una lunga fila indiana di numeri.
16×5×5=400.
Quindi entrano 400 numeri distinti.

**L'Output (120):**
Questi 400 numeri vengono trasformati in 120 nuove caratteristiche più astratte.

---

### Criterion

E' il nome convenzionale dato in pytorch per la loss function.
Criterion è la componenente matematica che calcola quanto il modello sta sbagliando
Prende l'output della tua rete (la previsione) e l'etichetta vera (la realtà), e restituisce un numero unico (la loss).

1) Se la loss è **alta** → il modello ha sbagliato di molto.
2) Se la loss è **bassa** → il modello ha quasi indovinato.

L'ottimizzatore (optimizer) userà questo numero per aggiornare i pesi e ridurre l'errore nel futuro.

Abbiamo 2 tipi principali di tipi di lossFunction in base al tipo di problema che devono risolvere:
1) Problemi di **Classificazione**
2) Problemi di **Regressione**

I tipi di funzioni per i problemi di **Classificazione** sono:
1) **nn.CrossEntropyLoss()**

**A che serve**: È la più usata per la classificazione Multi-classe (es. CIFAR-10, MNIST: classi 0-9).

**Come funziona:** Combina internamente una LogSoftmax e una NLLLoss. Si aspetta che la rete restituisca dei "logits" (valori grezzi non normalizzati) per ogni classe. Penalizza fortemente se la rete dà una probabilità bassa alla classe corretta.

**Esempio:** Riconoscere cifre scritte a mano.

2) **nn.BCEWithLogitsLoss() (Binary Cross Entropy)**

**A che serve:** Per la classificazione Binaria (solo due classi: 0 o 1, Sì o No).

**Come funziona:** È più stabile numericamente della semplice BCELoss. Si applica quando l'ultimo strato della rete è un unico neurone che deve dire "quanto è probabile che sia 1".

**Esempio:** Riconoscere se un'email è Spam (1) o Non Spam (0).

3) **nn.NLLLoss() (Negative Log Likelihood)**

**A che serve:** Simile alla CrossEntropy, ma si usa se la tua rete ha già un layer LogSoftmax come ultima uscita.

**Nota:** Spesso si preferisce usare direttamente CrossEntropyLoss (che include già la Softmax) per semplicità, ma NLLLoss è utile se devi manipolare le probabilità manualmente prima della loss

I tipi di funzioni per i problemi di **Regressione** sono:

1) **nn.MSELoss() (Mean Squared Error)**

**A che serve:** È lo standard per la regressione.

**Come funziona:** Calcola la media dei quadrati delle differenze tra previsione e realtà: (y− 
y
^
​	
 ) 
2
 .

**Caratteristica:** Poiché eleva al quadrato, penalizza enormemente gli errori grandi (gli outlier).

**Esempio:** Prevedere il prezzo di una casa in base ai metri quadri.

2) **nn.L1Loss() (Mean Absolute Error)**

A che serve: Regressione dove ci sono dati "sporchi" o outlier.

Come funziona: Calcola la differenza assoluta: ∣y− 
y
^
​	
 ∣.

**Caratteristica**: È più robusta agli outlier rispetto a MSE, perché l'errore cresce linearmente e non quadraticamente.

**Esempio**: Stimare il tempo di arrivo di un delivery (se un driver si ferma 1 ora, l'MSE sballerebbe tutto il training, L1 lo gestisce meglio).

3) **nn.SmoothL1Loss() (o Huber Loss)**

**A che serve:** Una via di mezzo tra MSE e L1.

**Come funziona**: Si comporta come MSE quando l'errore è piccolo (per convergere bene) e come L1 quando l'errore è grande (per non impazzire con gli outlier).

**Esempio**: Molto usata nel rilevamento oggetti (Object Detection, es. YOLO) per prevedere le coordinate dei box attorno agli oggetti.
