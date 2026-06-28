# Voice message source identifier

Voice message source identifier is a tool for classifying the origin of voice messages based on AAC encoding and container features. This project is associated with a research paper.


## Installation
1. Clone the repository
   ```bash   
   git clone git@github.com:mjkim100ku/Voice-Message-Source-Identifier.git
   cd Voice-Message-Source-Identifier/
   ```
2. Create and activate the conda environment (<https://docs.rapids.ai/install/>)
    ```bash    
    conda create -n rapids-25.04 -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.04 python=3.12 'cuda-version>=12.0,<=12.8'
    conda activate rapids-25.04
    ```
3. Install other dependencies
   ```bash   
   pip install -r requirements.txt
   ```


## Getting Started
- Download `ffmpeg.exe` and place it under `tools/ffmpeg/ffmpeg.exe` when running from source.
- Download `ffprobe.exe` and place it under `tools/ffmpeg/ffprobe.exe` when running from source.


## Dataset & Pre-built binary
You can download the dataset and **the Pre-built binary** from the following [Google Drive link](https://drive.google.com/drive/folders/1KXONLxWPNKDan4SpVGG2VjuiLVEcSdmc?usp=drive_link).


## Usage (cli)
You can classify voice message sources by specifying the folder path containing audio files as an argument.
```bash
python -m cli.classify_audio_files [path_to_folder_containing_audio_files]
```

The current default classifier is `trained_lr.pkl`, matching the representative LR model used in the manuscript revision. The original voting ensemble remains available:

```bash
python -m cli.classify_audio_files --model voting [path_to_folder_containing_audio_files]
```

## Revision experiments

The released tool loads packaged model files under `models/`. Manuscript revision experiments are kept separate under `experiments/` so they can be reviewed without mixing evaluation scripts with the public classifier workflow.

- `experiments/sota_baselines`: Jin et al. and Wang et al. comparison code, plus the curated SOTA result table used in the manuscript.
- `experiments/current_version_lr`: current app version validation scripts for the LR component and condition diagnostics.

Generated rerun outputs are ignored by Git. Curated manuscript snapshots are stored in each experiment folder under `paper_results/`.

## Demo video
[Demo video](https://youtu.be/Wqmtwa6Kt9Y)
