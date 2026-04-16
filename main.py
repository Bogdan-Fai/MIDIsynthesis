import argparse
from clearml import Task

from Services.midi_service import play_midi
from src.preprocess import preprocess
from src.train import train
from src.generate import generate


def setup_clearml(command_name: str):
    return Task.init(
        project_name="MIDIsynthesis",
        task_name=f"{command_name}_run",
        tags=["transformer", "midi"],
        auto_connect_arg_parser=True,
        auto_connect_frameworks=True,
        auto_resource_monitoring=True,
        auto_connect_streams=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        default="train",
        choices=["preprocess", "train", "generate", "last"],
    )
    parser.add_argument("--resume-task-id", type=str, default=None)
    parser.add_argument("--resume-artifact-name", type=str, default=None)

    args = parser.parse_args()
    command = args.command

    if command in ["preprocess", "train", "generate"]:
        task = setup_clearml(command)
        task.connect(args)
    else:
        task = None

    if command == "preprocess":
        preprocess("Data/MIDI", task=task)

    elif command == "train":
        train(
            task=task,
            resume_task_id=args.resume_task_id,
            resume_artifact_name=args.resume_artifact_name,
        )

    elif command == "generate":
        generate(task=task)

    elif command == "last":
        play_midi("Data/outputs/generated.mid")


if __name__ == "__main__":
    main()