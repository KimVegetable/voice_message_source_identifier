import argparse

from voiceid.inference import classify_folder, classify_files_streaming


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify voice message sources from a folder of audio files."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="test",
        help="Path to folder containing audio files.",
    )
    return parser.parse_args()


def main():
    # CLI entry point for batch classification.
    args = parse_args()
    # classify_folder(args.folder, verbose=True)
    classify_files_streaming(args.folder, verbose=True)
    


if __name__ == "__main__":
    main()
