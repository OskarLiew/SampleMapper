import matplotlib.pyplot as plt
from librosa.display import specshow
from IPython.display import Audio
import numpy as np


def plot_wav(waveform: np.ndarray, sr: int):
    length = waveform.shape[0] / sr
    time = np.linspace(0., length, waveform.shape[0])
    scaled_waveform = waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, scaled_waveform)
    plt.show()
    return Audio(waveform.T, rate=sr)


def plot_spectrogram(spec):
    specshow(spec)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
