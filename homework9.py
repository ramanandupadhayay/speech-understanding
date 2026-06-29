import numpy as np

def VAD(waveform, Fs):
    '''
    Voice Activity Detection based on energy
    '''
    frame_length = int(0.025 * Fs)  # 25 ms
    step = int(0.010 * Fs)          # 10 ms

    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    energies = []
    frames = []

    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_length]
        energy = np.sum(frame**2)
        energies.append(energy)
        frames.append(frame)

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    current_segment = []

    for i in range(num_frames):
        if energies[i] > threshold:
            current_segment.extend(frames[i])
        else:
            if len(current_segment) > 0:
                segments.append(np.array(current_segment))
                current_segment = []

    # last segment
    if len(current_segment) > 0:
        segments.append(np.array(current_segment))

    return segments


def segments_to_models(segments, Fs):
    '''
    Convert segments to spectral models
    '''
    models = []

    for seg in segments:
        # Pre-emphasis
        seg_pre = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])

        # framing
        frame_length = int(0.004 * Fs)  # 4 ms
        step = int(0.002 * Fs)          # 2 ms

        N = len(seg_pre)
        num_frames = 1 + (N - frame_length) // step

        frames = []
        for i in range(num_frames):
            start = i * step
            frames.append(seg_pre[start:start + frame_length])

        frames = np.array(frames)

        # FFT magnitude
        mstft = np.abs(np.fft.fft(frames, axis=1))

        # keep low-frequency half
        half = mstft.shape[1] // 2
        mstft = mstft[:, :half]

        # log spectrum
        mstft = np.maximum(mstft, 1e-6)
        log_spec = 20 * np.log10(mstft)

        # average spectrum → model
        model = np.mean(log_spec, axis=0)
        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize speech using cosine similarity
    '''
    # Get test segments
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, test_model in enumerate(test_models):
        best_label = None
        best_score = -1

        for y, model in enumerate(models):
            # cosine similarity
            dot = np.dot(model, test_model)
            norm = np.linalg.norm(model) * np.linalg.norm(test_model)
            sim = dot / (norm + 1e-10)

            sims[y, k] = sim

            if sim > best_score:
                best_score = sim
                best_label = labels[y]

        test_outputs.append(best_label)

    return sims, test_outputs