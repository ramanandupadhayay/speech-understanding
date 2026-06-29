import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop waveform into overlapping frames
    '''
    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):
    '''
    Magnitude FFT of each frame
    '''
    # FFT along each row
    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft


def mstft_to_spectrogram(mstft):
    '''
    Convert to decibel spectrogram
    '''
    # Avoid log(0) by flooring values
    floor = 0.001 * np.amax(mstft)
    mstft_safe = np.maximum(floor, mstft)

    # Convert to dB
    spectrogram = 20 * np.log10(mstft_safe)

    # Limit dynamic range to 60 dB
    max_val = np.amax(spectrogram)
    spectrogram = np.maximum(spectrogram, max_val - 60)

    return spectrogram