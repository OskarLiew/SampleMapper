from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).parent
MODEL_DIR = CONFIG_DIR

SAMPLE_RATE = 44100

SOUND_PATH = Path("E:/Ljud/Samples")
SOUND_PATH_LIST = list(SOUND_PATH.glob("**/*.wav"))
SPECTROGRAM_PATH = Path("E:/Data/sounds_spectrograms/spectrograms.pkl")

SAMPLE_MAPPER_APP_DIR = CONFIG_DIR.parent.parent / "sample_mapper"
RESULTS_DIR = SAMPLE_MAPPER_APP_DIR / "data"
SOUND_OUTPUT_PATH = SAMPLE_MAPPER_APP_DIR / "assets"
MODEL_PATH = CONFIG_DIR / "autoencoder_network.pth"

np.random.seed(100)
N_SAMPLES = 1000
NOT_DEMOS = np.array([i_p for i_p in enumerate(SOUND_PATH_LIST) if "Demos" not in str(i_p[1])])  # Remove copyrighted sounds
SAMPLE_IDX = np.random.choice(NOT_DEMOS[:, 0], N_SAMPLES, replace=False).astype(int)
FROM_PATHS = [SOUND_PATH_LIST[i].as_posix() for i in SAMPLE_IDX]
TO_PATHS = [(SOUND_OUTPUT_PATH / SOUND_PATH_LIST[i].name).as_posix() for i in SAMPLE_IDX]
