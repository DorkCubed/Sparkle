import torch
from torch.utils.data import DataLoader
from .dataset import PacketSequenceDataset
from sparkle.configs.config import Config
from sparkle.data_loader.tokenizer.tokenizer import Tokenizer


class DataModule:
    def __init__(self, device=None, config=None, base_path=None):
        print("Reached data module.")

        self.config = config if config is not None else Config()
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.base_path = base_path  # Base path for S3 remapping (train vs eval)
        print(f"Using device: {self.device}")
        print(f"Using manifest: {self.config.manifest_path}")
        if self.base_path:
            print(f"Using custom base path: {self.base_path}")

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
            chunk_size=self.config.chunk_size,
            device=self.device,
            base_path=self.base_path,
        )

    def _init_dataloader(self):
        # For CUDA, we need to use num_workers=0 to avoid multiprocessing issues
        # with CUDA initialization in worker processes
        num_workers = (
            0 if (torch.cuda.is_available() and self.device.type == "cuda") else 4
        )

        if num_workers == 0:
            # When using CUDA, don't use multiprocessing
            return DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        else:
            # When using CPU, we can use multiprocessing with custom collate
            return DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
            )

    def get_batches(self):
        """Generator that yields batches with tensors moved to the correct device"""
        for batch in self.train_loader:
            packet_sequences, field_positions, header_positions, entries = batch

            # Move tensors to device if CUDA is available
            if torch.cuda.is_available() and self.device.type == "cuda":
                packet_sequences = packet_sequences.to(self.device, non_blocking=True)
                field_positions = field_positions.to(self.device, non_blocking=True)
                header_positions = header_positions.to(self.device, non_blocking=True)

            yield packet_sequences, field_positions, header_positions, entries

    def get_batches_with_length(self):
        """Returns a generator and length for progress bars"""
        return self.get_batches(), len(self.train_loader)

    # TODO: definitely not pythonic, update later
    def get_loader(self):
        return self.train_loader

    # TODO: should we make tokenizer a property? Is a getter unnecessary?
    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device


def test_data_loader(data_loader, device=None, num_batches=2):
    """
    Test a data loader by iterating over it and printing out the first num_batches.

    Args:
        data_loader (DataLoader): A PyTorch DataLoader object.
        device (torch.device): Device to move tensors to for testing
        num_batches (int): The number of batches to print out. Defaults to 2.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (
        packet_sequences,
        field_position,
        header_position,
        packet_seq_file_name,
    ) in enumerate(data_loader):
        print(f"\nBatch {i + 1}")

        print(f"Packet Sequences shape: {packet_sequences.shape}")
        print(f"Field Position shape: {field_position.shape}")
        print(f"Header Position shape: {header_position.shape}")
        print(f"File Name: {packet_seq_file_name}")
        print(f"Device: {packet_sequences.device}")

        print("First packet sequence tokens:", packet_sequences[0, :50])

        assert isinstance(packet_sequences, torch.Tensor)
        assert isinstance(field_position, torch.Tensor)
        assert isinstance(header_position, torch.Tensor)

        if i + 1 >= num_batches:
            break


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = DataModule(device=device)
    train_loader = data_module.get_loader()
    test_data_loader(train_loader, device=device, num_batches=2)
