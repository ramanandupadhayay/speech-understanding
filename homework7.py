import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord.
    '''
    duration = 0.5
    t = np.arange(0, duration, 1/Fs)

    # Frequencies for major chord
    f_root = f
    f_third = f * 2**(4/12)   # 4 semitones up
    f_fifth = f * 2**(7/12)   # 7 semitones up

    # Generate tones
    x_root = np.sin(2 * np.pi * f_root * t)
    x_third = np.sin(2 * np.pi * f_third * t)
    x_fifth = np.sin(2 * np.pi * f_fifth * t)

    # Combine and normalize
    x = (x_root + x_third + x_fifth) / 3

    return x


def dft_matrix(N):
    '''
    Create DFT matrix of size NxN
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies
    '''
    N = len(x)

    # FFT
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1/Fs)

    # Only take positive frequencies
    half = N // 2
    X = np.abs(X[:half])
    freqs = freqs[:half]

    # Find top 3 peaks
    indices = np.argsort(X)[-3:]
    top_freqs = freqs[indices]

    # Sort frequencies
    f1, f2, f3 = np.sort(top_freqs)

    return f1, f2, f3