import sys

from src.preprocess import preprocess
from src.train import train
# from src.generate import generate


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py preprocess")
        print("  python main.py train")
        print("  python main.py generate")
        return

    command = sys.argv[1].lower()

    if command == "preprocess":
        preprocess("Data/MIDI")
    elif command == "train":
        train()
    # elif command == "generate":
    #     run_generate()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: preprocess, train, generate")


if __name__ == "__main__":
    main()