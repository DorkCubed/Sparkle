import os

class Config:
    def __init__(self):
        bucket = 'netml-s3-bucket'
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

        self.packet_folder = os.path.join(parent_dir, "dataset", "packets")
        self.header_folder = os.path.join(parent_dir, "dataset", "headers")
        self.fields_folder = os.path.join(parent_dir, "dataset", "fields")
        self.direction_folder = os.path.join(parent_dir, "dataset", "directions")

        self.tokenizer_path = os.path.join(current_dir, "tokenizer", "vocab.txt")

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
