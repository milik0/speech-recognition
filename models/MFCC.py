import torchaudio
from torchaudio import transforms

def _MFCC(waveform, sample_rate):
    transform = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc = transform(waveform)
    return mfcc
