import torchaudio
import torch
from torchaudio.datasets import LIBRISPEECH
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.MFCC import _MFCC
from models.MLP import MLP

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

def train_model(batch_size, num_workers, num_epochs=1, save_path='model_checkpoint.pth'):

    train_loader = manage_data(batch_size, num_workers)
    
    # 13 => Corresponds to the shape of the num_frames
    # 128 => is arbitrary 
    # 29 => corresponds to a-z, " ", "'" characters corresponding to the possible outputs
    model = MLP(input_size=13, hidden_size=128, output_size=29)
    model.train(True)
    
    # Better optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    char2idx = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '")}
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (waveforms, sample_rates, transcripts, speaker_ids, chapter_ids, utterance_ids) in enumerate(train_loader):
            
            batch_log_probs = []
            batch_targets = []
            batch_input_lengths = []
            batch_target_lengths = []
            
            # Prepare batch data
            for i, (waveform, sample_rate) in enumerate(zip(waveforms, sample_rates)):
                mfcc = _MFCC(waveform, sample_rate)
                mfcc = mfcc.squeeze(0).transpose(0, 1)  # [time_frames, 13]
                
                target = torch.tensor([char2idx[c] for c in transcripts[i].lower() if c in char2idx])
                
                # CRITICAL: Check sequence lengths
                if mfcc.size(0) <= len(target):
                    print(f"Skipping sample {i}: input_len={mfcc.size(0)}, target_len={len(target)}")
                    continue
                
                logits = model(mfcc)  # [time_frames, output_size]
                log_probs = F.log_softmax(logits, dim=1)
                
                batch_log_probs.append(log_probs)
                batch_targets.append(target)
                batch_input_lengths.append(log_probs.size(0))
                batch_target_lengths.append(len(target))
            
            if len(batch_log_probs) == 0:
                continue
            
            # Pad sequences to same length
            max_input_len = max(batch_input_lengths)
            padded_log_probs = torch.zeros(max_input_len, len(batch_log_probs), 29)
            
            for i, log_probs in enumerate(batch_log_probs):
                padded_log_probs[:log_probs.size(0), i, :] = log_probs
            
            # Concatenate targets
            concatenated_targets = torch.cat(batch_targets)
            input_lengths = torch.tensor(batch_input_lengths, dtype=torch.long)
            target_lengths = torch.tensor(batch_target_lengths, dtype=torch.long)
            
            # Single backward pass per batch
            optimizer.zero_grad()
            loss = model.loss(padded_log_probs, concatenated_targets, input_lengths, target_lengths)
            
            if torch.isnan(loss):
                print(f"NaN loss detected at batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Avg Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model