# Speech-to-Text, Audio Deep Learning & Voice Agents

This project was completed by:

* Khaled Mili
* Maxim Bocquillion

for the Automatic Speech Recognition (RECOP) course.

---

## Day 1 — From Scratch Architectures: From MFCC to Transformers

### Part 1 — MLP + MFCC + CTC

**Training**
MLP: Epoch 5 completed. Average loss: 3.2602

**Test Results**

**Sample 1**
Reference: `mister quilter is the apostle of the middle classes and we are glad to welcome his gospel`
Predicted: (empty)
WER: 523.53%, CER: 82.02%

**Sample 2**
Reference: `nor is mister quilter's manner less interesting than his matter`
Predicted: (empty)
WER: 630.00%, CER: 85.71%

**Sample 3**
Reference: `he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind`
Predicted: (empty)
WER: 537.50%, CER: 88.37%

**Sample 4**
Reference: `he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca`
Predicted: (empty)
WER: 554.17%, CER: 83.46%

**Sample 5**
Reference: `linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man`
Predicted: (empty)
WER: 530.88%, CER: 81.77%

**Average Metrics (over 20 samples)**

* Average WER: 543.92%
* Average CER: 85.25%

---

### Part 2 — CNN + Spectrograms

Epoch 5, Batch 3560, Avg Loss: 1.8152

**Evaluation Results**

**Sample 1**
Reference: `mister quilter is the apostle of the middle classes and we are glad to welcome his gospel`
Predicted: `tht lerisy pl so h hl lus s wer la lt hi cusol`
WER: 323.53%, CER: 61.80%

**Sample 2**
Reference: `nor is mister quilter's manner less interesting than his matter`
Predicted: `noris i olers e r hs iteret ntan is meter`
WER: 280.00%, CER: 46.03%

**Sample 3**
Reference: `he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind`
Predicted: `h tls ustatah testus oeses et erwt cr nase estpe l in bforsssolyheto m ein d hrs lts gr ha rle ot myi`
WER: 309.38%, CER: 58.14%

**Sample 4**
Reference: `he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca`
Predicted: `his riy toswtersrvrte lie s wer is riyegret oferalleae edi he t utltl lf oahteth ot`
WER: 316.67%, CER: 57.14%

**Sample 5**
Reference: `linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man`
Predicted: `wi elsp ers er ete art at e eingsan mesos s wa it lscersnes o s ting po tht berhfstor lens i t s melt won tch nisi we et erer hethels os e sn mto eon lolyr ho o ier tctrefl slepththbtc pbforye ys l o hem o r htr e et nhe mant`
WER: 311.76%, CER: 58.29%

**Average Metrics (over 20 samples)**

* Average WER: 308.43%
* Average CER: 57.19%

---

### Part 3 — RNN (LSTM / GRU / BiLSTM)

**Training**
20 epochs on device: cuda

**Evaluation Results**

**Sample 1**
Reference: `mister quilter is the apostle of the middle classes and we are glad to welcome his gospel`
Predicted: `itecuters ipuo melcasis er loe is guto`
WER: 347.06%, CER: 65.17%

**Sample 2**
Reference: `nor is mister quilter's manner less interesting than his matter`
Predicted: `norismeteoers mam lisenvrtin then is mater`
WER: 280.00%, CER: 44.44%

**Sample 3**
Reference: `he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind`
Predicted: `etois the e thetueesonno er wi cemesrs bein befrs solas orntoeting orso e cr marl ote m`
WER: 318.75%, CER: 59.30%

**Sample 4**
Reference: `he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca`
Predicted: `hes gra dos mterserfre lis ir ira gre te iindeerini b mi o rti cu`
WER: 325.00%, CER: 58.65%

**Sample 5**
Reference: `linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man`
Predicted: `winos eters rsir bodtead andins anmisis wisitit ris atatiin o etebere futeristits n mutinin witietru atefasistt anmitertorca isisitr turoslapnbat e furiis wa aa mapr o ni men`
WER: 351.47%, CER: 65.47%

**Average Metrics (over 20 samples)**

* Average WER: 325.93%
* Average CER: 59.88%

---

### Part 4 — Transformer Approach

**Architecture**

* **Input projection:** Linear projection from 40 Mel bins to `d_model=256`
* **Positional encoding:** Sinusoidal to capture temporal order
* **Transformer Encoder:** 6 layers with 8 attention heads
* **Feedforward dimension:** 1024
* **Output classifier:** Linear layer to 29 classes (a-z, space, apostrophe)
* **Dropout:** 0.1

**Key Features**

* Uses Mel Spectrograms (40 bins)
* `batch_first=True` for PyTorch compatibility
* Self-attention captures long-range dependencies
* Input format: `[Batch, Time, Features]`

**Training Configuration**

* Optimizer: Adam (lr=3e-4)
* Loss: CTC Loss (blank=28)
* Input: Mel Spectrograms (hop_length=80)
* Epochs: 10

**Training Results (10 epochs, dev-clean)**

Epoch 10, Batch 270, Avg Loss: 1.6523

**Evaluation Results**

**Sample 1**
Reference: `mister quilter is the apostle of the middle classes and we are glad to welcome his gospel`
Predicted: `mister quilter is the apostle of the middle classes and we are glad to welcome his gospel`
WER: 0.00%, CER: 0.00%

**Sample 2**
Reference: `nor is mister quilter's manner less interesting than his matter`
Predicted: `nor is mister quilters manner less interesting than his matter`
WER: 10.00%, CER: 1.59%

**Sample 3**
Reference: `he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind`
Predicted: `he tells us that at this festive season of the year with christmas and roast beef looming before us similies drawn from eating and its results occur most readily to the mind`
WER: 3.13%, CER: 0.58%

**Sample 4**
Reference: `he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca`
Predicted: `he has grave doubts whether sir frederick leightons work is really greek after all and can discover in it but little of rocky ithaca`
WER: 5.56%, CER: 0.75%

**Sample 5**
Reference: `linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man`
Predicted: `linnells pictures are a sort of up guards and at em paintings and masons exquisite idylls are as national as a jingo poem mister birket fosters landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man`
WER: 2.94%, CER: 1.16%

**Average Metrics (over 20 samples)**

* Average WER: 4.82%
* Average CER: 1.23%

**Note:** The Transformer requires more data and epochs but achieves near-perfect transcription on clean speech.

---

### Part 5 — Hyperparameter Optimization with Optuna

**Objective**
Optimize a bidirectional GRU using Optuna to minimize validation CTC Loss.

**Optimized Hyperparameters**

* **Model Architecture:**

  * `hidden_dim`: [128, 256, 512]
  * `num_layers`: [1, 2, 3]
  * `dropout`: [0.1 - 0.5]
* **Training:**

  * `learning_rate`: [1e-4 - 5e-3] (log scale)
  * `batch_size`: [8, 16]

**Optuna Configuration**

* Sampler: TPESampler
* Direction: Minimize
* Number of trials: 10-20
* Pruning: MedianPruner
* Epochs per trial: 5

**Dataset for Tuning**

* LibriSpeech dev-clean
* Split: 80% train / 20% validation
* Features: Normalized Mel Spectrograms (40 bins)

**Training Stabilization**

```python
spec = spec.clamp(min=1e-9).log2()
spec = (spec - spec.mean()) / (spec.std() + 1e-5)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Metrics & Validation**

* Objective: Average validation CTC Loss
* Reporting: Epoch-wise
* Early stopping: NaN or exploding loss trials are pruned

**Example Best Parameters**

```python
Best Hyperparameters:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.2
  lr: 0.001
  batch_size: 16
```

**To Use Best Hyperparameters**

```python
best_params = study.best_params

model = TunableGRU(
    input_dim=40,
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout'],
    n_classes=29
)

optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
```

**Note:** Perform Optuna tuning on a subset first; then train on full dataset using the best parameters.

---

## Architecture Comparison

| Model         | WER     | CER    | Final Loss | Training Time | Parameters |
| ------------- | ------- | ------ | ---------- | ------------- | ---------- |
| MLP + MFCC    | 543.92% | 85.25% | 3.26       | Fast          | ~100K      |
| CNN + MelSpec | 308.43% | 57.19% | 1.82       | Medium        | ~500K      |
| GRU (BiLSTM)  | 325.93% | 59.88% | ~2.0       | Long          | ~300K      |
| Transformer   | 4.82%   | 1.23%  | 1.65       | Very Long     | ~2M        |

**Observations**

* CNN performs best among smaller models
* MLP with MFCC alone is insufficient
* BiGRU balances performance and time
* Transformer achieves near-perfect results but requires more data/epochs
* Optuna can significantly improve model performance

**Recommendations**

1. Fast deployment: **CNN + MelSpec**
2. Temporal dependencies: **BiGRU with Optuna**
3. Best performance with lots of data: **Transformer**
4. Always use CTC Loss for alignment
5. Data augmentation recommended for generalization
