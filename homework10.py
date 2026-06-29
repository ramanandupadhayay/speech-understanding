import numpy as np
import torch
import torch.nn as nn

def get_features(waveform, Fs):
    # ---------- Pre-emphasis ----------
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # ---------- Spectrogram ----------
    frame_len = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)        # 2 ms

    N = len(waveform)
    num_frames = 1 + (N - frame_len) // step

    frames = []
    for i in range(num_frames):
        start = i * step
        frames.append(waveform[start:start + frame_len])
    frames = np.array(frames)

    # FFT and magnitude
    mstft = np.abs(np.fft.fft(frames, axis=1))

    # keep low-frequency half
    half = mstft.shape[1] // 2
    features = mstft[:, :half]

    # ---------- VAD (for labels) ----------
    vad_len = int(0.025 * Fs)  # 25 ms
    vad_step = int(0.010 * Fs) # 10 ms

    num_vad = 1 + (N - vad_len) // vad_step

    energies = []
    for i in range(num_vad):
        start = i * vad_step
        frame = waveform[start:start + vad_len]
        energies.append(np.sum(frame**2))

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    labels = np.zeros(num_frames, dtype=int)

    current_label = 1
    idx = 0

    for i in range(num_vad):
        if energies[i] > threshold:
            # assign this label to 5 frames (approx mapping)
            for _ in range(5):
                if idx < num_frames:
                    labels[idx] = current_label
                    idx += 1
        else:
            current_label += 1

    return features, labels


def train_neuralnet(features, labels, iterations):
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    NFEATS = features.shape[1]
    NLABELS = int(np.max(labels)) + 1

    # Model: LayerNorm + Linear
    model = nn.Sequential(
        nn.LayerNorm(NFEATS),
        nn.Linear(NFEATS, NLABELS)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    lossvalues = []

    for _ in range(iterations):
        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_fn(outputs, y)

        loss.backward()
        optimizer.step()

        lossvalues.append(loss.item())

    return model, np.array(lossvalues)


def test_neuralnet(model, features):
    X = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)

    return probs.detach().numpy()