import os
from sparkle.utils import get_project_root


class Config:
    def __init__(self):
        self.bucket = 'netml-s3-bucket'
        project_root = get_project_root()
        current_dir = os.path.dirname(__file__)
        self.split_folder = "split"
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        self.parents = [f"{self.bucket}/Working_folder/input_aws/Wireshark_Sample_PCAPs/split/", f"{self.bucket}/Working_folder/input_aws/"]

        # might potentially take time
        self.fetcher = None
        self.files = None

        self.tokenizer_path = os.path.join(parent_dir, "data_loader", "tokenizer", "vocab.txt")
        # self.manifest_path = os.path.join(parent_dir, "manifest", "manifest.json")
        # self.manifest_path = os.path.join(parent_dir, "manifest", "test_manifest.json")
        self.manifest_path = os.path.join(project_root, "manifest", "manifest.json")
        self.batch_size = 1
        self.shuffle = False

        self.vocab_size = 262
        self.embed_dim = 256  # 768
        self.num_heads = 8  # 12
        self.num_layers = 4  # 6
        self.dropout = 0.2  # 0.1
        self.max_flow_length = 512
        self.mask_prob = 0.15
        self.num_epochs = 1
        self.max_len = 578  # 512
        self.chunk_size = 2 # earlier batch_size_1
        self.learning_rate = 0.001

    def initialize_data_fetcher(self):
        """Call this when you actually need the data"""
        if self.fetcher is None:
            from src.sparkle.data_loader.scripts.s3_utils import S3DataFetcher
            self.fetcher = S3DataFetcher(self.bucket)
            self.files = self.fetcher.list_split_objects(self.parents, split=self.split_folder)