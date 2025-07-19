import argparse
import sys
from scripts.training import train

# run file python main.py train --dataset datasets/literature/shakespear.txt --hidden_size 250 --epochs 1 --sample_size 500

def main():
    parser = argparse.ArgumentParser(description="LSTM Model Project Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    
    train_parser = subparsers.add_parser("train", help="Train the LSTM model")
    train_parser.add_argument('--dataset', type=str, default='datasets/code/oop_dataset_1000_lines.txt', help='Path to dataset')
    train_parser.add_argument('--hidden_size', type=int, default=250, help='Hidden size of LSTM')
    train_parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    train_parser.add_argument('--sample_size', type=int, default=500, help='Sample size for text generation')

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()