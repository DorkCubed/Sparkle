import torch
from torch.utils.data import DataLoader
from .dataset import PacketSequenceDataset
from sparkle.configs.config import Config
from sparkle.data_loader.tokenizer.tokenizer import Tokenizer

class DataModule:
    def __init__(self):
        print("Reached data module.")

        self.config = Config()
        self.tokenizer = self._load_tokenizer()

        self.dataset = self._init_dataset()
        print("Loaded dataset.")

        self.train_loader = self._init_dataloader()
        print("Init dataloader.")

        print("Initiated data module.")

    def _load_tokenizer(self):
        return Tokenizer(vocab_file=self.config.tokenizer_path)

    def _init_dataset(self):
        return PacketSequenceDataset(
            config=self.config,
            manifest_path=self.config.manifest_path,
            tokenizer=self.tokenizer,
            chunk_size=self.config.chunk_size
        )

    def _init_dataloader(self):
        num_workers = 4
        return DataLoader(self.dataset, 
                          batch_size=self.config.batch_size, 
                          shuffle=False, 
                          num_workers=num_workers,
                          pin_memory = True,
                          prefetch_factor=2,
                          persistent_workers=True
                         )

    # TODO: definitely not pythonic, update later
    def get_loader(self):
        return self.train_loader

    # TODO: should we make tokenizer a property? Is a getter unnecessary?
    def get_tokenizer(self):
        return self.tokenizer

def test_data_loader(data_loader, num_batches=2):
    """
    Test a data loader by iterating over it and printing out the first num_batches.

    Args:
        data_loader (DataLoader): A PyTorch DataLoader object.
        num_batches (int): The number of batches to print out. Defaults to 2.
    """
    for i, (packet_sequences, field_position, header_position, packet_seq_file_name) in enumerate(data_loader):
        print(f"\nBatch {i+1}")
        print(f"Packet Sequences shape: {packet_sequences.shape}")
        print(f"Field Position shape: {field_position.shape}")
        print(f"Header Position shape: {header_position.shape}")
        print(f"File Name: {packet_seq_file_name}")

        print("First packet sequence tokens:", packet_sequences[0, :50])

        assert isinstance(packet_sequences, torch.Tensor)
        assert isinstance(field_position, torch.Tensor)
        assert isinstance(header_position, torch.Tensor)

        if i + 1 >= num_batches:
            break


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = DataModule()
    train_loader = data_module.get_loader()
    test_data_loader(train_loader, num_batches=2)
