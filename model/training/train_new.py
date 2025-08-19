import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BERTModel
from Input_Tokenizer import Tokenizer
from dataset import PacketSequenceDataset
import sagemaker_containers
import socket


def train(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    vocab = {}
    with open(args.vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            token, token_id = line.strip().split('\t')
            vocab[token] = int(token_id)

    # Initialize tokenizer
    tokenizer = Tokenizer(vocab_file=args.vocab_file)

    # Create dataset and data loader
    dataset = PacketSequenceDataset(
        packet_seq_dir=args.packet_seq_dir,
        field_pos_dir=args.field_pos_dir,
        header_pos_dir=args.header_pos_dir,
        direction_dir=args.direction_dir,
        tokenizer=tokenizer
    )
    train_loader = DataLoader(
        dataset, batch_size=args.chunk_size, shuffle=True)

    # Initialize model
    model = BERTModel(
        vocab_size=args.vocab_size,
        vocab=args.vocab_file,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_flow_length=args.max_flow_length,
        mask_prob=args.mask_prob
    ).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for packet_sequences, field_pos, header_pos, direction in train_loader:
            packet_sequences = packet_sequences.to(device)
            field_pos = field_pos.to(device)
            header_pos = header_pos.to(device)
            direction = direction.to(device)

            optimizer.zero_grad()
            loss = model(packet_sequences, field_pos, header_pos, direction)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{args.num_epochs}, Loss: {total_loss/len(train_loader)}")

        # Save checkpoints
        if args.current_host == args.hosts[0]:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.model_dir, f'checkpoint_epoch_{epoch}.pt'))

    # Final model save
    if args.current_host == args.hosts[0]:
        torch.save(model.state_dict(), os.path.join(
            args.model_dir, 'model.pth'))

    # Save the model
    # torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # sagemaker specific arguments
    parser.add_argument('--hosts', type=list,
                        default=sagemaker_containers.read_hosts())
    parser.add_argument('--current-host', type=str,
                        default=sagemaker_containers.current_host())

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--packet-seq-dir', type=str,
                        default=os.environ['SM_CHANNEL_PACKET_SEQ'])
    parser.add_argument('--field-pos-dir', type=str,
                        default=os.environ['SM_CHANNEL_FIELD_POS'])
    parser.add_argument('--header-pos-dir', type=str,
                        default=os.environ['SM_CHANNEL_HEADER_POS'])
    parser.add_argument('--direction-dir', type=str,
                        default=os.environ['SM_CHANNEL_DIRECTION'])
    parser.add_argument('--vocab-file', type=str,
                        default=os.environ['SM_CHANNEL_VOCAB'])

    # Training Parameters
    parser.add_argument('--vocab-size', type=int, default=261)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--num-heads', type=int, default=12)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-flow-length', type=int, default=510)
    parser.add_argument('--mask-prob', type=float, default=0.15)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    args = parser.parse_args()
    train(args)
