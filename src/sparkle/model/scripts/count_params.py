import torch
import logging
from sparkle.model.embedding import PacketEmbedding, FlowEmbedding
from sparkle.model.flow_encoder import FlowLevelEncoder
from sparkle.model.packet_encoder import PacketLevelEncoder
from sparkle.configs.config import Config
from datetime import datetime
import os

# Configure logging
os.makedirs('logs', exist_ok=True)
log_file = f'logs/count_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def count_model_parameters():
    logger.info("Counting parameters...")
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    vocab = {}
    with open(config.tokenizer_path, 'r', encoding='utf-8') as f:
        for line in f:
            token, token_id = line.strip().split('\t')
            vocab[token] = int(token_id)
    logger.info(f"Vocabulary size: {len(vocab)}")

    packet_embedding = PacketEmbedding(config.vocab_size, max_len=config.max_len, embed_dim=config.embed_dim,
                                       dropout=config.dropout).to(device)
    packet_encoder = PacketLevelEncoder(config.vocab_size, config.embed_dim, config.max_len, config.num_heads,
                                        config.num_layers, config.dropout).to(device)
    flow_embedding = FlowEmbedding(config.embed_dim, config.max_flow_length, config.dropout, vocab).to(device)
    flow_encoder = FlowLevelEncoder(config.embed_dim, config.num_layers, config.num_heads, config.dropout, vocab,
                                    config.max_flow_length, config.mask_prob).to(device)

    logger.info("Models initialized successfully.")

    all_models = [packet_embedding, packet_encoder, flow_embedding, flow_encoder]
    total_trainable_params = sum(
        p.numel() for model in all_models for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 50)
    logger.info(
        f"Total trainable parameters: {total_trainable_params:,} (approx. {total_trainable_params / 1_000_000:.2f}M)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    count_model_parameters()