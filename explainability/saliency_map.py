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
    highlight_color='red',
    base_color='#bfbfbd',
    figsize=(14, 4),
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


def plot_spectrogram_with_saliency_modulated(
    spectrogram,
    sr,
    hop_length,
    saliency,
    audio,
    threshold=0.3,
    highlight_cmap='inferno',
    base_cmap='gray_r',
    figsize=(20, 6),
    title="Spectrogram with Saliency-Modulated Color",
):
    """
    Plot a spectrogram where full vertical time slices are colored
    if the mean saliency in that time window (from raw waveform) exceeds threshold.
    
    Parameters:
    - spectrogram: np.ndarray, shape (freq_bins, time_frames)
    - sr: int, sample rate
    - hop_length: int, STFT hop length
    - saliency: 1D np.ndarray of shape (samples,)
    - audio: 1D np.ndarray of shape (samples,)
    - threshold: float, saliency threshold per time frame
    """
    
    # Normalize spectrogram
    norm_spec = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-6)
    
    freq_bins, time_frames = spectrogram.shape
    total_samples = len(audio)
    
    # Compute saliency per time frame
    saliency_per_frame = []
    for frame_idx in range(time_frames):
        start = frame_idx * hop_length
        end = min(start + hop_length, total_samples)
        mean_sal = np.mean(saliency[start:end]) if end > start else 0
        saliency_per_frame.append(mean_sal)
    saliency_per_frame = np.array(saliency_per_frame)
    
    # Get colormaps
    base_cm = plt.get_cmap(base_cmap)
    highlight_cm = plt.get_cmap(highlight_cmap)
    
    # Build color image
    colored_image = np.zeros((*spectrogram.shape, 4))  # RGBA
    for j in range(time_frames):
        is_salient = saliency_per_frame[j] >= threshold
        cmap = highlight_cm if is_salient else base_cm
        for i in range(freq_bins):
            val = norm_spec[i, j]
            colored_image[i, j] = cmap(val)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    duration = time_frames * hop_length / sr
    extent = [0, duration, 0, sr // 2]
    
    ax.imshow(colored_image, aspect='auto', origin='lower', extent=extent)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()