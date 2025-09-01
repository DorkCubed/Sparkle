import torch
from sparkle.model.scripts.train import PacketLevelTrainer
from sparkle.model.scripts.train import ExperimentRunner
from pathlib import Path

def test_process_encodings():
    from sparkle.data_loader.data_loader import DataModule
    from sparkle.model.embedding import PacketEmbedding, FlowEmbedding
    from sparkle.model.flow_encoder import FlowLevelEncoder
    from sparkle.model.packet_encoder import PacketLevelEncoder
    from sparkle.configs.config import Config

    runner = ExperimentRunner()
    vocab = runner.load_vocab()
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init modules
    packet_embedding = PacketEmbedding(config.vocab_size, config.max_len, config.embed_dim, config.dropout).to(device)
    packet_encoder = PacketLevelEncoder(config.vocab_size, config.embed_dim, config.max_len,
                                        config.num_heads, config.num_layers, config.dropout).to(device)
    flow_embedding = FlowEmbedding(config.embed_dim, config.max_flow_length, config.dropout, vocab).to(device)
    flow_encoder = FlowLevelEncoder(config.embed_dim, config.num_layers, config.num_heads, config.dropout,
                                    config.max_flow_length, config.mask_prob).to(device)

    trainer = PacketLevelTrainer(packet_embedding, packet_encoder, flow_embedding, flow_encoder)

    # load data
    data_module = DataModule()
    train_loader = data_module.get_loader()

    packet_sequences, field_position, header_position, entry = next(iter(train_loader))

    mlm_loss, sfbo_loss, encoded_packets_mean = trainer.packet_encoder(
        packet_sequences.squeeze(0).to(device),
        field_pos=field_position.squeeze(0).to(device),
        header_pos=header_position.squeeze(0).to(device)
    )

    # feed encodings into process_encodings
    mpm_loss = trainer.process_encodings([encoded_packets_mean], entry)

    print("Final returned mpm_loss:", mpm_loss)

if __name__ == "__main__":
    test_process_encodings()
