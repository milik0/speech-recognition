import torchaudio
import torch
from torchaudio.datasets import LIBRISPEECH
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.MFCC import _MFCC
from models.MLP import MLP
from models.CNN import CNN
from models.GRU import GRU
from models.Transformers import Transformer

# Model selection parameter - change this to 'MLP', 'CNN', 'GRU', or 'Transformer'
MODEL_TYPE = 'CNN'  # Options: 'MLP', 'CNN', 'GRU', 'Transformer'

def collate_fn(batch):
    """Custom collate function to handle variable-length audio"""
    waveforms = [item[0] for item in batch]
    sample_rates = [item[1] for item in batch]
    transcripts = [item[2] for item in batch]
    speaker_ids = [item[3] for item in batch]
    chapter_ids = [item[4] for item in batch]
    utterance_ids = [item[5] for item in batch]
    
    return waveforms, sample_rates, transcripts, speaker_ids, chapter_ids, utterance_ids

def manage_data(batch_size, num_workers):
    train_dataset = LIBRISPEECH("./data", url="train-clean-100", download=True)
    print("Data Downloaded !")
    
    print(f"Dataset size: {len(train_dataset)} samples")
    print(f"First sample shape: {train_dataset[0][0].shape}")

    # Create DataLoader for batch processing
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader

def train_model(batch_size, num_workers, num_epochs=5, save_path='model_checkpoint.pth', model_type=MODEL_TYPE):

    # -------------------------------
    # GPU / MPS / CPU management
    # -------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_loader = manage_data(batch_size, num_workers)
    
    # Initialize model based on type
    if model_type == 'MLP':
        model = MLP(input_size=13, hidden_size=128, output_size=29)
        print("Using MLP")
    elif model_type == 'CNN':
        model = CNN(n_classes=29, n_mels=40)
        print("Using CNN")
    elif model_type == 'GRU':
        model = GRU(input_dim=40, hidden_dim=128, num_layers=2, n_classes=29)
        print("Using GRU")
    elif model_type == 'Transformer':
        model = Transformer(input_dim=40, d_model=256, nhead=8, num_layers=6, n_classes=29)
        print("Using Transformer")
    else:
        raise ValueError("Unknown model_type")

    # Move model to GPU/MPS/CPU
    model = model.to(device)
    model.train(True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    ctc_loss = torch.nn.CTCLoss(blank=28, zero_infinity=True)

    char2idx = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '")}

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (waveforms, sample_rates, transcripts, _, _, _) in enumerate(train_loader):

            batch_log_probs = []
            batch_targets = []
            batch_input_lengths = []
            batch_target_lengths = []

            for i, (waveform, sample_rate) in enumerate(zip(waveforms, sample_rates)):

                # Move waveform to device
                waveform = waveform.to(device)

                # -------------------------------
                # Feature extraction (on device)
                # -------------------------------
                if model_type == 'MLP':
                    features = _MFCC(waveform, sample_rate)  # should return on same device
                    features = features.squeeze(0).transpose(0, 1)

                elif model_type == 'CNN':
                    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=80
                    ).to(device)

                    features = mel_spec_transform(waveform)
                    features = features.clamp(min=1e-9).log2()
                    features = features.unsqueeze(0)

                elif model_type == 'GRU':
                    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=80
                    ).to(device)

                    features = mel_spec_transform(waveform)
                    features = features.clamp(min=1e-9).log2()
                    # GRU expects [B, T, F] -> transpose
                    features = features.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T, 40]

                elif model_type == 'Transformer':
                    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_mels=40,
                        n_fft=400,
                        hop_length=80
                    ).to(device)

                    features = mel_spec_transform(waveform)
                    features = features.clamp(min=1e-9).log2()
                    # Transformer expects [B, T, F]
                    features = features.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, T, 40]

                # Target on device
                target = torch.tensor(
                    [char2idx[c] for c in transcripts[i].lower() if c in char2idx],
                    dtype=torch.long,
                    device=device
                )

                # Length checks
                if model_type == 'MLP':
                    input_len = features.size(0)
                elif model_type == 'CNN':
                    input_len = features.size(-1) // 8
                elif model_type == 'GRU':
                    input_len = features.size(1)  # Time dimension for GRU
                elif model_type == 'Transformer':
                    input_len = features.size(1)  # Time dimension for Transformer

                if input_len <= len(target):
                    print(f"Skipping sample {i}: input_len={input_len}, target_len={len(target)}")
                    continue

                # Forward
                logits = model(features)
                if model_type == 'CNN':
                    logits = logits.squeeze(0)
                elif model_type == 'GRU':
                    logits = logits.squeeze(0)
                elif model_type == 'Transformer':
                    logits = logits.squeeze(0)

                log_probs = F.log_softmax(logits, dim=1)

                batch_log_probs.append(log_probs)
                batch_targets.append(target)
                batch_input_lengths.append(log_probs.size(0))
                batch_target_lengths.append(len(target))

            if len(batch_log_probs) == 0:
                continue

            # -------------------------------
            # Pad sequences (on device)
            # -------------------------------
            max_input_len = max(batch_input_lengths)
            padded_log_probs = torch.zeros(
                max_input_len, len(batch_log_probs), 29,
                device=device
            )

            for i, log_probs in enumerate(batch_log_probs):
                padded_log_probs[:log_probs.size(0), i, :] = log_probs

            concatenated_targets = torch.cat(batch_targets)
            input_lengths = torch.tensor(batch_input_lengths, dtype=torch.long, device=device)
            target_lengths = torch.tensor(batch_target_lengths, dtype=torch.long, device=device)

            optimizer.zero_grad()
            loss = ctc_loss(padded_log_probs, concatenated_targets, input_lengths, target_lengths)

            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Avg Loss: {total_loss/num_batches:.4f}")

        print(f"\nEpoch {epoch+1} completed. Avg loss = {total_loss/num_batches:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model
