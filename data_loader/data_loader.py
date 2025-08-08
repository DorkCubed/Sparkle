import torch
from torch.utils.data import DataLoader
from .dataset import PacketSequenceDataset
from .config import Config
from tokenizer.tokenizer import Tokenizer

class DataModule:
    def __init__(self):
        self.config = Config()
        self.tokenizer = self._load_tokenizer()
        self.dataset = self._init_dataset()
        self.train_loader = self._init_dataloader()

    def _load_tokenizer(self):
        return Tokenizer(vocab_file=self.config.tokenizer_path)

    def _init_dataset(self):
        return PacketSequenceDataset(config.packet_folder, config.header_folder, config.fields_folder, self.tokenizer, chunk_size=self.config.chunk_size)

    def _init_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.chunk_size, shuffle=False)

    def _get_loader(self):
        return self.train_loader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = DataModule()

    config = data_module.config

    tokenizer = data_module.tokenizer

    dataset = PacketSequenceDataset(config.packet_folder, config.header_folder, config.fields_folder, tokenizer, chunk_size=32)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)







