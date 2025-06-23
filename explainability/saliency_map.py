import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import librosa.display
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap



def compute_saliency_waveform(model, input_values, target_class=None, segment_length=5, overlap=0.2, target_sr=16000):
    # Ensure model and input are on the same device
    device = next(model.parameters()).device

    # Create a new leaf variable with requires_grad=True
    input_values = input_values.clone().detach().to(device)
    input_values.requires_grad_(True)

    # Get audio dimensions
    batch_size, num_segments, segment_samples = input_values.shape
    step_samples = int(segment_length * target_sr * (1 - overlap))
    
    # Calculate the exact audio length based on segments and overlap
    # Formula: (num_segments - 1) * step_samples + segment_samples
    audio_length = (num_segments - 1) * step_samples + segment_samples
    actual_duration = audio_length / target_sr  # Actual duration in seconds

    # Forward pass
    model.eval()
    with torch.set_grad_enabled(True):
        output = model.speech_only_forward(input_values)

        # If target class not specified, use predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass - compute gradient for target class
        model.zero_grad()
        output[0, target_class].backward()

        # Get gradients and compute saliency
        gradients = input_values.grad
        if gradients is None:
            raise RuntimeError("No gradients computed. Check if your model supports gradient computation.")

        saliency = gradients.abs().squeeze(0).cpu().detach().numpy()

    # Create accurate time axis
    t = np.linspace(0, actual_duration, audio_length)

    # Initialize full-length saliency array
    full_saliency = np.zeros(audio_length)
    overlap_counts = np.zeros(audio_length)
    original_waveform = np.zeros(audio_length)

    # Map segment saliency and waveform back to original audio positions
    input_values_cpu = input_values.detach().cpu()
    for i in range(num_segments):
        start_sample = i * step_samples
        end_sample = start_sample + segment_samples
        end_sample = min(end_sample, audio_length)
        seg_length = end_sample - start_sample

        # Process saliency
        segment_saliency = saliency[i][:seg_length] if i < len(saliency) else np.zeros(seg_length)
        full_saliency[start_sample:end_sample] += segment_saliency
        overlap_counts[start_sample:end_sample] += 1

        # Process waveform
        segment = input_values_cpu[0, i, :seg_length].numpy()
        original_waveform[start_sample:end_sample] += segment

    # Normalize by overlap counts
    overlap_counts[overlap_counts == 0] = 1
    smoothed_saliency = full_saliency / overlap_counts
    original_waveform = original_waveform / overlap_counts

    # Additional smoothing (25ms window)
    window_size = int(0.025 * target_sr)
    if window_size > 0:
        window = np.ones(window_size) / window_size
        smoothed_saliency = np.convolve(smoothed_saliency, window, mode='same')

    return {
        'waveform': original_waveform,
        'saliency': smoothed_saliency,
        'time': t,
        'target_class': target_class,
        'class_name': model.label_rev_map.get(target_class, str(target_class)),
        'actual_duration': actual_duration  # For debugging
    }

def plot_saliency_waveform(saliency_data, pauses=None):

    """
    Plot waveform, saliency map, and pauses if available

    Args:
        saliency_data: Dictionary containing saliency information
        pauses: List of pause tuples (start, end, _, _, _, _, mark) or None
    """
    plt.figure(figsize=(14, 4))
    ax = plt.gca()

    # Plot waveform and saliency
    plt.plot(saliency_data['time'], saliency_data['waveform'], label="Waveform", alpha=0.7)
    plt.plot(saliency_data['time'], saliency_data['saliency'], label="Saliency", color="red", alpha=0.7)

    # Plot pauses if available
    if pauses:
        # Create dummy lines for legend
        ax.plot([], [], color="#e33d19", linestyle="-", linewidth=2, label="Informative Pause")
        ax.plot([], [], color="#e33d19", linestyle="--", linewidth=2, label="Natural Pause")

        # Find maximum value for scaling pause boxes
        max_val = max(np.max(saliency_data['waveform']), np.max(saliency_data['saliency']))

        for start, end, *_, mark in pauses:
            linestyle = "-" if mark else "--"
            ax.plot([start, start, end, end, start],
                    [0, max_val, max_val, 0, 0],
                    color="#e33d19",
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=0.3)

            # Fill the pause area
            ax.fill_between([start, end],
                          [0, 0],
                          [max_val, max_val],
                          color="#e33d19",
                          alpha=0.1)

    plt.legend()
    plt.title(f"Waveform and Saliency (Class: {saliency_data['class_name']})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude / Saliency")
    plt.tight_layout()
    plt.show()



def plot_waveform_with_saliency(
    waveform,
    saliency,
    time,
    threshold=0.3,
    highlight_color='orange',
    base_color='#bfbfbd',
    figsize=(14, 5),
    title="Waveform Colored by Saliency",
    xlabel="Time (s)",
    ylabel="Amplitude"
):
    """
    Plot waveform with saliency-based color highlights.

    Parameters:
    - waveform: np.ndarray, audio waveform
    - saliency: np.ndarray, same shape as waveform, importance values (0 to 1)
    - time: np.ndarray, time axis corresponding to waveform
    - threshold: float, minimum saliency to trigger color change
    - highlight_color: str, color for high-saliency regions
    - base_color: str, color for low-saliency regions
    - figsize: tuple, figure size
    - title: str, plot title
    - xlabel: str, x-axis label
    - ylabel: str, y-axis label
    """

    # Normalize saliency and threshold
    saliency = np.clip(saliency, 0, 1)
    saliency = (saliency - threshold) / (1 - threshold)
    saliency = np.clip(saliency, 0, 1)

    # Prepare segments
    points = np.array([time, waveform]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    saliency_avg = (saliency[:-1] + saliency[1:]) / 2

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("saliency_map", [base_color, highlight_color])

    # Create colored line segments
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), linewidth=1.0)
    lc.set_array(saliency_avg)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()



def plot_spectrogram_with_saliency(
    spectrogram,
    sr,
    hop_length,
    saliency,
    threshold=0.3,
    highlight_cmap='inferno',
    base_cmap='gray_r',
    figsize=(14, 6),
    title="Spectrogram with Saliency Overlay",
    alpha=0.8
):
    """
    Plot a spectrogram with important regions (based on saliency) highlighted.

    Parameters:
    - spectrogram: np.ndarray, shape (freq_bins, time_frames)
    - sr: int, sample rate
    - hop_length: int, hop length used in STFT
    - saliency: 1D or 2D np.ndarray (time-aligned with spectrogram frames)
    - threshold: float, saliency values below this will not be highlighted
    - highlight_cmap: str, colormap for important regions
    - base_cmap: str, colormap for the default grayscale spectrogram
    - figsize: tuple, size of the figure
    - title: str, plot title
    - alpha: float, transparency of the saliency overlay
    """

    # Normalize and reshape saliency
    saliency = np.clip(saliency, 0, 1)
    if saliency.ndim == 1:
        saliency = np.tile(saliency, (spectrogram.shape[0], 1))  # Expand to 2D
    elif saliency.shape != spectrogram.shape:
        raise ValueError("Saliency shape must match spectrogram shape or be 1D of time dimension.")

    # Create masks
    mask = saliency >= threshold
    masked_saliency = np.where(mask, saliency, 0)

    # Time & frequency axes
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.specshow(
        librosa.power_to_db(spectrogram, ref=np.max),
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap=base_cmap,
        ax=ax
    )

    # Overlay saliency in color
    ax.imshow(
        masked_saliency,
        cmap=highlight_cmap,
        alpha=alpha,
        aspect='auto',
        extent=[0, spectrogram.shape[1] * hop_length / sr, 0, sr // 2],
        origin='lower'
    )

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.colorbar(ax.images[-1], ax=ax, label='Saliency')
    plt.tight_layout()
    plt.show()
