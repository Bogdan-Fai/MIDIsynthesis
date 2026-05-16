import argparse
from clearml import Task

from Services.midi_service import play_midi
from src.preprocess import preprocess
from src.train import train
from src.generate import generate


def setup_clearml(command_name: str):
    return Task.init(
        project_name="MIDIsynthesis",
        task_name=f"{command_name}_run_with_regularization",
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible generation")

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
        generate(path="../../Downloads/midi_transformer_final (2).pt" ,task=task, seed=None)
        # generate(task=task)

    elif command == "last":
        play_midi("Data/outputs/generated_20260514_161733.mid")


if __name__ == "__main__":
    main()