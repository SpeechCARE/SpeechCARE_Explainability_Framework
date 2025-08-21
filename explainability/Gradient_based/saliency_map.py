import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from scipy.signal import find_peaks, convolve
from scipy import interpolate


def normalize_saliency(saliency, threshold):
    return np.clip((saliency - threshold) / (1 - threshold), 0, 1)

def interpolate_saliency(saliency_data, target_times,min_saliency=0.6):
    # saliency = smooth_saliency_gaps(saliency_data['saliency'], window_size=10, overlap=2, high_thresh=0.4, gap_fill_thresh=0.5)
    # saliency = np.where(saliency_data['saliency'] <min_saliency , min_saliency, saliency_data['saliency'])
    saliency = saliency_data['saliency']
    return interpolate.interp1d(
        saliency_data['time'],
        saliency,
        kind='linear',
        bounds_error=False,
        fill_value=0
    )(target_times)


def load_and_resample(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # Shape: [samples]


def compute_raw_saliency(model, input_values, target_class=None):
    device = next(model.parameters()).device
    input_values = input_values.clone().detach().to(device)
    input_values.requires_grad_(True)

    model.eval()
    with torch.set_grad_enabled(True):
        output = model.speech_only_forward(input_values)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, target_class].backward()
        gradients = input_values.grad

        if gradients is None:
            raise RuntimeError("Gradients not computed.")

        saliency = gradients.abs().squeeze(0).cpu().detach().numpy()  # Shape: [num_segments, segment_samples]

    return saliency, input_values.cpu().detach().squeeze(0), target_class

def reconstruct_saliency_and_waveform(saliency, waveform_segments, segment_length, overlap, sample_rate, original_length=None):
    segment_samples = waveform_segments.shape[1]
    step_samples = int(segment_samples * (1 - overlap))
    num_segments = waveform_segments.shape[0]

    # Estimate full length or use original
    if original_length is None:
        total_length = (num_segments - 1) * step_samples + segment_samples
    else:
        total_length = original_length

    time_axis = np.linspace(0, total_length / sample_rate, total_length)

    full_saliency = np.zeros(total_length)
    full_waveform = np.zeros(total_length)
    overlap_counts = np.zeros(total_length)

    for i in range(num_segments):
        start = i * step_samples
        end = start + segment_samples
        if end > total_length:
            end = total_length
            seg_len = end - start
            full_saliency[start:end] += saliency[i][:seg_len]
            full_waveform[start:end] += waveform_segments[i][:seg_len].numpy()
            overlap_counts[start:end] += 1
        else:
            full_saliency[start:end] += saliency[i][:segment_samples]
            full_waveform[start:end] += waveform_segments[i][:segment_samples].numpy()
            overlap_counts[start:end] += 1

    overlap_counts[overlap_counts == 0] = 1
    avg_saliency = full_saliency / overlap_counts
    avg_waveform = full_waveform / overlap_counts

    return avg_saliency, avg_waveform, time_axis


def smooth_and_normalize_saliency(saliency, sample_rate):
    epsilon = 1e-10

    # Step 1: Adaptive clipping based on signal variation
    iqr = np.percentile(saliency, 75) - np.percentile(saliency, 25)
    upper_bound = np.percentile(saliency, 75) + 1.5 * iqr
    saliency = np.clip(saliency, None, upper_bound)

    # Step 2: Estimate dynamic smoothing window based on saliency peak spacing
    peaks, _ = find_peaks(saliency, height=np.percentile(saliency, 70))
    if len(peaks) >= 2:
        avg_distance = np.mean(np.diff(peaks))  # in samples
        smooth_window = int(min(max(avg_distance / 2, 0.01 * sample_rate), 0.1 * sample_rate))
    else:
        smooth_window = int(0.025 * sample_rate)  # fallback (25ms)

    kernel = np.ones(smooth_window) / smooth_window
    smoothed = convolve(saliency, kernel, mode='same')

    # Step 3: Local contrast-aware enhancement (optional)
    # Emphasize parts that remain strong even after smoothing
    local_std = np.std(smoothed)
    adaptive_threshold = smoothed.mean() + 0.5 * local_std
    boost_mask = smoothed > adaptive_threshold
    boosted = np.where(boost_mask, smoothed * 1.1, smoothed)  # light boost

    # Step 4: Normalize
    normalized = (boosted - boosted.min()) / (boosted.max() - boosted.min() + epsilon)

    return normalized

# def compute_saliency_waveform(model, audio_path, input_segments, target_class=None, segment_length=5, overlap=0.2, target_sr=16000):
#     waveform = load_and_resample(audio_path, target_sr)
#     original_length = len(waveform)
#     raw_saliency, raw_segments, target_class = compute_raw_saliency(model, input_segments, target_class)
#     saliency, waveform, time = reconstruct_saliency_and_waveform(
#         raw_saliency, raw_segments, segment_length, overlap, target_sr, original_length=original_length
#     )

#     smoothed_saliency = smooth_and_normalize_saliency(saliency, target_sr)

#     return {
#         'waveform': waveform,
#         'saliency': smoothed_saliency,
#         'time': time,
#         'target_class': target_class,
#         'class_name': model.label_rev_map.get(target_class, str(target_class)),
#         'actual_duration': original_length / target_sr
#     }


def _segment_windows(num_segments, segment_length_sec, overlap, sample_rate, total_samples):
    """
    Compute [start,end) sample indices and [start_sec, end_sec] for each segment
    given N segments, segment length, overlap, and the total waveform length.
    """
    L = int(round(segment_length_sec * sample_rate))                # segment length in samples
    step = int(round(L * (1.0 - overlap)))                          # hop size
    windows = []
    windows_sec = []

    for i in range(num_segments):
        start = i * step
        end = start + L
        # clip to waveform bounds
        start_clipped = max(0, min(start, total_samples))
        end_clipped = max(0, min(end, total_samples))
        windows.append((start_clipped, end_clipped))
        windows_sec.append((start_clipped / sample_rate, end_clipped / sample_rate))
    return windows, windows_sec


def compute_saliency_waveform(model, audio_path, input_segments, target_class=None,
                              segment_length=5, overlap=0.2, target_sr=16000):
    """
    Returns:
      - waveform: np.array, original-length waveform (resampled)
      - saliency: np.array, per-sample normalized saliency in [0,1]
      - time: np.array, time axis (seconds) for 'waveform'/'saliency'
      - target_class: int
      - class_name: str
      - actual_duration: float seconds
      - segment_saliency_mean: np.array of shape [num_segments], mean saliency per segment window
      - segment_windows_sec: list of (start_sec, end_sec) for each segment
    """
    # Load / resample
    waveform = load_and_resample(audio_path, target_sr)  # 1D np.array
    original_length = len(waveform)

    # Raw gradients w.r.t. raw segmented input (your existing function)
    raw_saliency, raw_segments, target_class = compute_raw_saliency(
        model, input_segments, target_class
    )

    # Align raw per-sample saliency back to the continuous waveform timeline (your existing function)
    saliency, waveform, time = reconstruct_saliency_and_waveform(
        raw_saliency, raw_segments, segment_length, overlap, target_sr, original_length=original_length
    )

    # Smooth + normalize to [0,1]
    smoothed_saliency = smooth_and_normalize_saliency(saliency, target_sr)

    # ----- NEW: per-segment saliency means -----
    # Determine number of segments from input_segments shape: [num_segments, seq_length] or [1, num_segments, seq_length]
    if input_segments.ndim == 2:
        num_segments = input_segments.shape[0]
    elif input_segments.ndim == 3:
        num_segments = input_segments.shape[1]
    else:
        raise ValueError("input_segments must be [num_segments, seq_length] or [1, num_segments, seq_length].")

    # Compute segment sample windows and average saliency per segment
    windows, windows_sec = _segment_windows(
        num_segments=num_segments,
        segment_length_sec=segment_length,
        overlap=overlap,
        sample_rate=target_sr,
        total_samples=original_length
    )

    segment_means = []
    for (start, end) in windows:
        if end > start:
            segment_means.append(float(np.mean(smoothed_saliency[start:end])))
        else:
            # Empty (fully clipped) window: define as 0.0
            segment_means.append(0.0)
    segment_means = np.asarray(segment_means, dtype=float)

    return {
        'waveform': waveform,
        'saliency': smoothed_saliency,
        'time': time,
        'target_class': target_class,
        'class_name': model.label_rev_map.get(target_class, str(target_class)),
        'actual_duration': original_length / target_sr,
        'segment_saliency_mean': segment_means,      # [num_segments]
        'segment_windows_sec': windows_sec           # [(start_sec, end_sec)] * num_segments
    }