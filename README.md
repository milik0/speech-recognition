d # 🎧 Sujet global — Speech-To-Text, Audio Deep Learning & Agents Vocaux

Ce TP a été effectué par :
- Khaled Mili
- Maxim Bocquillion

pour le cours de Reconnaissance Automatique de la Parole (RECOP)

## 🗓️ Jour 1 — Architecture From Scratch : de MFCC à Transformers

### MLP + MFCC + CTC

**Training**

MLP : Epoch 5 completed. Average loss: 3.2602

**Test**

Sample 1:
  Reference: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
  Predicted:                 
  WER: 523.53%, CER: 82.02%

Sample 2:
  Reference: nor is mister quilter's manner less interesting than his matter
  Predicted:               
  WER: 630.00%, CER: 85.71%

Sample 3:
  Reference: he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind
  Predicted:                     
  WER: 537.50%, CER: 88.37%

Sample 4:
  Reference: he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca
  Predicted:                       
  WER: 554.17%, CER: 83.46%

Sample 5:
  Reference: linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man
  Predicted:                                    e                              
  WER: 530.88%, CER: 81.77%

================================================================================
AVERAGE METRICS (over 20 samples)
  Average WER: 543.92%
  Average CER: 85.25%
================================================================================

(5.43915519855178, 0.8525168635140311)

### Partie 2 — CNN + Spectrogrammes

Epoch 5, Batch 3560, Avg Loss: 1.8152

================================================================================
EVALUATION RESULTS
================================================================================

Sample 1:
  Reference:  mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
  Predicted:  tht lerisy pl so h hl lus s wer la lt hi cusol
  WER: 323.53%, CER: 61.80%

Sample 2:
  Reference:  nor is mister quilter's manner less interesting than his matter
  Predicted:  noris i olers e r hs iteret ntan is meter 
  WER: 280.00%, CER: 46.03%

Sample 3:
  Reference:  he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind
  Predicted:   h tls ustatah testus oeses et erwt cr nase estpe l in bforsssolyheto m ein d hrs lts gr ha rle ot myi 
  WER: 309.38%, CER: 58.14%

Sample 4:
  Reference:  he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca
  Predicted:  his riy toswtersrvrte lie s wer is riyegret oferalleae edi he t utltl lf oahteth ot
  WER: 316.67%, CER: 57.14%

Sample 5:
  Reference:  linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man
  Predicted:  wi elsp ers er ete  art at e eingsan mesos s wa it lscersnes o s ting po tht berhfstor lens i t s melt won tch nisi we et erer  hethels os e sn mto eon lolyr  ho o ier tctrefl slepththbtc pbforye ys l o hem o r htr e et  nhe mant
  WER: 311.76%, CER: 58.29%

================================================================================
AVERAGE METRICS (over 20 samples)
  Average WER: 308.43%
  Average CER: 57.19%
================================================================================

(3.0843232000759473, 0.5718563758910706)

### Partie 3 — RNN (LSTM / GRU / BiLSTM)

On 20 epoch

Evaluating on device: cuda

================================================================================
EVALUATION RESULTS
================================================================================

Sample 1:
  Reference:  mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
  Predicted:  itecuters ipuo melcasis er  loe is guto
  WER: 347.06%, CER: 65.17%

Sample 2:
  Reference:  nor is mister quilter's manner less interesting than his matter
  Predicted:  norismeteoers mam lisenvrtin then is mater
  WER: 280.00%, CER: 44.44%

Sample 3:
  Reference:  he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind
  Predicted:  etois the e thetueesonno er wi cemesrs bein befrs solas orntoeting orso e cr marl ote m
  WER: 318.75%, CER: 59.30%

Sample 4:
  Reference:  he has grave doubts whether sir frederick leighton's work is really greek after all and can discover in it but little of rocky ithaca
  Predicted:  hes gra dos mterserfre lis ir ira gre te iindeerini b mi o rti   cu
  WER: 325.00%, CER: 58.65%

Sample 5:
  Reference:  linnell's pictures are a sort of up guards and at em paintings and mason's exquisite idylls are as national as a jingo poem mister birket foster's landscapes smile at one much in the same way that mister carker used to flash his teeth and mister john collier gives his sitter a cheerful slap on the back before he says like a shampooer in a turkish bath next man
  Predicted:  winos eters rsir  bodtead andins anmisis wisitit ris atatiin o etebere futeristits  n mutinin witietru atefasistt anmitertorca isisitr turoslapnbat e furiis wa aa mapr o ni men
  WER: 351.47%, CER: 65.47%

================================================================================
AVERAGE METRICS (over 20 samples)
  Average WER: 325.93%
  Average CER: 59.88%
================================================================================

(3.259282575603885, 0.5988340362371821)

### Partie 4 — Approche Transformers
