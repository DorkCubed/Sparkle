import torch
from torch.utils.data import DataLoader
from .dataset import PacketSequenceDataset
from .config import Config
from .tokenizer.tokenizer import Tokenizer

class DataModule:
    def __init__(self):
        self.config = Config()
        self.tokenizer = self._load_tokenizer()
        self.dataset = self._init_dataset()
        self.train_loader = self._init_dataloader()

    def _load_tokenizer(self):
        return Tokenizer(vocab_file=self.config.tokenizer_path)

    def _init_dataset(self):
        return PacketSequenceDataset(self.config.packet_folder, self.config.header_folder, self.config.fields_folder, self.tokenizer, chunk_size=self.config.chunk_size)

    def _init_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.chunk_size, shuffle=False)

    def _get_loader(self):
        return self.train_loader


def test_data_loader(data_loader, num_batches=2):
    for i, (packet_sequences, field_position, header_position, packet_seq_file_name) in enumerate(data_loader):
        print(f"\nBatch {i+1}")
        print(f"Packet Sequences shape: {packet_sequences.shape}")
        print(f"Field Position shape: {field_position.shape}")
        print(f"Header Position shape: {header_position.shape}")
        print(f"File Name: {packet_seq_file_name}")

        # Optional: Inspect first few token IDs
        print("First packet sequence tokens:", packet_sequences[0, :50])

        # Sanity checks
        assert isinstance(packet_sequences, torch.Tensor), "packet_sequences is not a tensor"
        assert isinstance(field_position, torch.Tensor), "field_position is not a tensor"
        assert isinstance(header_position, torch.Tensor), "header_position is not a tensor"

        if i + 1 >= num_batches:
            break



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = DataModule()

    config = data_module.config

    tokenizer = data_module.tokenizer

    dataset = PacketSequenceDataset(config.packet_folder, config.header_folder, config.fields_folder, tokenizer, chunk_size=32)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    test_data_loader(train_loader, 3)






