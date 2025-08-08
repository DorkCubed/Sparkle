import os

class Config:
    def __init__(self):
        bucket = 'netml-s3-bucket'
        self.packet_folder = 'Working_folder/input_aws/split/packets'
        self.header_folder = 'Working_folder/input_aws/split/headers'
        self.fields_folder = 'Working_folder/input_aws/split/fields'
        self.direction_folder = 'Working_folder/input_aws/split/direction'

        self.tokenizer_path = os.path.join("tokenizer", "vocab.txt")

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
        self.chunk_size = 2
