import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
from models.MFCC import _MFCC
from models.MLP import MLP
from models.CNN import CNN
from models.GRU import GRU
from models.Transformers import Transformer
from pipeline.train import collate_fn
import Levenshtein

def decode_predictions(log_probs, idx2char, blank_idx=28):
    predictions = torch.argmax(log_probs, dim=1)

    decoded = []
    prev_char = None
    for pred in predictions:
        pred_idx = pred.item()
        if pred_idx != blank_idx and pred_idx != prev_char:
            decoded.append(idx2char[pred_idx])
        prev_char = pred_idx
    return ''.join(decoded)

def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
    return distance / len(ref_words)

def calculate_cer(reference, hypothesis):
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)


# ============================================================
#   E V A L U A T I O N    W I T H    G P U / M P S / C P U
# ============================================================

def evaluate_model(model, dataloader, model_type='MLP', device=None, num_samples=None):

    # -----------------------------
    # Device selection
    # -----------------------------
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"\nEvaluating on device: {device}")

    model = model.to(device)
    model.eval()

    chars = "abcdefghijklmnopqrstuvwxyz '"
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}

    total_wer = 0.0
    total_cer = 0.0
    num_processed = 0

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    with torch.no_grad():
        for batch_idx, (waveforms, sample_rates, transcripts, _, _, _) in enumerate(dataloader):

            for i, (waveform, sample_rate, transcript) in enumerate(
                zip(waveforms, sample_rates, transcripts)
            ):

                waveform = waveform.to(device)

                # ---------------------------------------------------
                # Feature extraction (GPU where possible)
                # ---------------------------------------------------
                if model_type == 'MLP':

                    features = _MFCC(waveform, sample_rate)  # your MFCC fn
                    features = features.squeeze(0).transpose(0, 1)
                    features = features.to(device)

                elif model_type == 'CNN':

                    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=160
                    )

                    # Some torchaudio kernels cannot run on MPS/GPU → move after transform
                    mel_spec_transform = mel_spec_transform.to(device)

                    features = mel_spec_transform(waveform)
                    features = features.clamp(min=1e-9).log2()
                    features = features.unsqueeze(0).to(device)

                elif model_type == 'GRU':

                    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=160
                    )

                    mel_spec_transform = mel_spec_transform.to(device)

                    features = mel_spec_transform(waveform)
                    features = features.clamp(min=1e-9).log2()
                    # GRU expects [B, T, F]
                    features = features.squeeze(0).transpose(0, 1).unsqueeze(0).to(device)  # [1, T, 40]

                elif model_type == 'Transformer':

                    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=160
                    )

                    mel_spec_transform = mel_spec_transform.to(device)

                    features = mel_spec_transform(waveform)
                    features = features.clamp(min=1e-9).log2()
                    # Transformer expects [B, T, F]
                    features = features.squeeze(0).transpose(0, 1).unsqueeze(0).to(device)  # [1, T, 40]

                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                # Forward → logits
                logits = model(features)
                if model_type == 'CNN':
                    logits = logits.squeeze(0)
                elif model_type == 'GRU':
                    logits = logits.squeeze(0)
                elif model_type == 'Transformer':
                    logits = logits.squeeze(0)

                log_probs = F.log_softmax(logits, dim=1)

                # Decode text
                predicted_text = decode_predictions(log_probs, idx2char)
                reference_text = transcript.lower()

                # Metrics
                wer = calculate_wer(reference_text, predicted_text)
                cer = calculate_cer(reference_text, predicted_text)

                total_wer += wer
                total_cer += cer
                num_processed += 1

                # Print first few samples
                if num_processed <= 5:
                    print(f"\nSample {num_processed}:")
                    print(f"  Reference:  {reference_text}")
                    print(f"  Predicted:  {predicted_text}")
                    print(f"  WER: {wer:.2%}, CER: {cer:.2%}")

                if num_samples and num_processed >= num_samples:
                    break

            if num_samples and num_processed >= num_samples:
                break

    avg_wer = total_wer / num_processed if num_processed > 0 else 0
    avg_cer = total_cer / num_processed if num_processed > 0 else 0

    print("\n" + "="*80)
    print(f"AVERAGE METRICS (over {num_processed} samples)")
    print(f"  Average WER: {avg_wer:.2%}")
    print(f"  Average CER: {avg_cer:.2%}")
    print("="*80 + "\n")

    return avg_wer, avg_cer


# ============================================================
#     L O A D   &   E V A L U A T E
# ============================================================

def load_and_evaluate(model_path=None, model_type='MLP', batch_size=8, num_workers=2, num_samples=50):

    # -----------------------------
    # Device selection
    # -----------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing device: {device}")

    # Dataset
    eval_dataset = LIBRISPEECH("./data", url="dev-clean", download=True)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # Model init
    print(f"Initializing {model_type} model...")
    if model_type == 'MLP':
        model = MLP(input_size=13, hidden_size=128, output_size=29)
    elif model_type == 'CNN':
        model = CNN(n_classes=29, n_mels=40)
    elif model_type == 'GRU':
        model = GRU(input_dim=40, hidden_dim=128, num_layers=2, n_classes=29)
    elif model_type == 'Transformer':
        model = Transformer(input_dim=40, d_model=256, nhead=8, num_layers=6, n_classes=29)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    if model_path:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: No model_path provided → evaluating random weights")

    # Evaluate
    return evaluate_model(
        model,
        eval_loader,
        model_type=model_type,
        device=device,
        num_samples=num_samples
    )

if __name__ == "__main__":
    load_and_evaluate(
        model_path='model_checkpoint.pth',
        model_type='GRU',
        num_samples=20
    )
