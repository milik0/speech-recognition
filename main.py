import os
from pipeline.train import train_model

def load_data(batch_size=8, num_workers=2):
    print("Downloading data")

    os.makedirs("data", exist_ok=True)
    # Use 'dev-clean' for testing, 'train-clean-100' for training
    train_model(batch_size, num_workers)

if __name__ == "__main__":
    load_data()
