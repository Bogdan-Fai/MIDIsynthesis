import sys
from clearml import Task, task

from src.preprocess import preprocess
from src.train import train
from src.generate import generate

def setup_clearml(command_name: str):
    task = Task.init(
        project_name="MIDIsynthesis",
        task_name=f"{command_name}_run",
        tags=["transformer", "midi"],
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True,
        auto_resource_monitoring=True,
        auto_connect_streams=True,
    )
    return task

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py preprocess")
        print("  python main.py train")
        print("  python main.py generate")
        return

    command = sys.argv[1].lower()
    task = setup_clearml(command)

    if command == "preprocess":
        preprocess("Data/MIDI")
    elif command == "train":
        train(task=task)
    elif command == "generate":
        generate(task=task)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: preprocess, train, generate")


if __name__ == "__main__":
    main()