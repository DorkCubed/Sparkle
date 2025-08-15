import torch
from torch.utils.data import DataLoader
from .dataset import PacketSequenceDataset
from .config import Config
from .tokenizer.tokenizer import Tokenizer
from scripts.s3_utils import S3DataFetcher

class DataModule:
    def __init__(self):
        self.config = Config()
        self.tokenizer = self._load_tokenizer()
        self.s3_fetcher = S3DataFetcher(bucket_name=self.config.bucket_name)
        self.files = self.s3_fetcher.list_split_objects(self.config.parents, self.config.split)
        self.dataset = self._init_dataset()
        self.train_loader = self._init_dataloader()

    def _load_tokenizer(self):
        return Tokenizer(vocab_file=self.config.tokenizer_path)

    def _init_dataset(self):
        return PacketSequenceDataset(
            config=self.config,
            s3_fetcher=self.s3_fetcher,
            files=self.files,
            tokenizer=self.tokenizer,
            chunk_size=self.config.chunk_size
        )

    def _init_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

    def get_loader(self):
        return self.train_loader


def test_data_loader(data_loader, num_batches=2):
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
    test_data_loader(train_loader, 3)
