import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
from models.MFCC import _MFCC
from models.MLP import MLP
from models.CNN import CNN
from pipeline.train import collate_fn
import Levenshtein

def decode_predictions(log_probs, idx2char, blank_idx=28):
    """Decode CTC predictions using greedy decoding"""
    # Get the most likely character at each timestep
    predictions = torch.argmax(log_probs, dim=1)
    
    # Remove consecutive duplicates and blanks
    decoded = []
    prev_char = None
    for pred in predictions:
        pred_idx = pred.item()
        if pred_idx != blank_idx and pred_idx != prev_char:
            decoded.append(idx2char[pred_idx])
        prev_char = pred_idx
    
    return ''.join(decoded)

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
    wer = distance / len(ref_words)
    return wer

def calculate_cer(reference, hypothesis):
    """Calculate Character Error Rate"""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    distance = Levenshtein.distance(reference, hypothesis)
    cer = distance / len(reference)
    return cer

def evaluate_model(model, dataloader, model_type='MLP', device='cpu', num_samples=None):
    """Evaluate the model on a dataset"""
    model.eval()
    model.to(device)
    
    # Character mapping
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
            for i, (waveform, sample_rate, transcript) in enumerate(zip(waveforms, sample_rates, transcripts)):
                # Process audio based on model type
                if model_type == 'MLP':
                    features = _MFCC(waveform, sample_rate)
                    features = features.squeeze(0).transpose(0, 1).to(device)  # [time_frames, 13]
                elif model_type == 'CNN':
                    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=160
                    )
                    features = mel_spec_transform(waveform)
                    features = features.log2()
                    features = features.unsqueeze(0).to(device)  # [1, 1, n_mels, time]
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                # Get predictions
                logits = model(features)
                if model_type == 'CNN':
                    logits = logits.squeeze(0)  # [time_frames', output_size]
                log_probs = F.log_softmax(logits, dim=1)
                
                # Decode
                predicted_text = decode_predictions(log_probs, idx2char)
                reference_text = transcript.lower()
                
                # Calculate metrics
                wer = calculate_wer(reference_text, predicted_text)
                cer = calculate_cer(reference_text, predicted_text)
                
                total_wer += wer
                total_cer += cer
                num_processed += 1
                
                # Print first few examples
                if num_processed <= 5:
                    print(f"\nSample {num_processed}:")
                    print(f"  Reference: {reference_text}")
                    print(f"  Predicted: {predicted_text}")
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

def load_and_evaluate(model_path=None, model_type='MLP', batch_size=8, num_workers=2, num_samples=50):
    """Load a trained model and evaluate it
    
    Args:
        model_path: Path to saved model weights
        model_type: Model architecture ('MLP', 'CNN', 'VGG11', 'VGG13', 'VGG16', 'VGG19')
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        num_samples: Number of samples to evaluate
    """
    # Load dataset
    eval_dataset = LIBRISPEECH("./data", url="dev-clean", download=True)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize model based on type
    print(f"Initializing {model_type} model...")
    if model_type == 'MLP':
        model = MLP(input_size=13, hidden_size=128, output_size=29)
    elif model_type == 'CNN':
        model = CNN(n_classes=29, n_mels=40)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load trained weights if provided
    if model_path:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("Warning: Evaluating untrained model (random weights)")
    
    # Evaluate
    wer, cer = evaluate_model(model, eval_loader, model_type=model_type, num_samples=num_samples)
    
    return wer, cer

if __name__ == "__main__":
    load_and_evaluate(
        model_path='model_checkpoint.pth',
        model_type='CNN',
        num_samples=20
    )
