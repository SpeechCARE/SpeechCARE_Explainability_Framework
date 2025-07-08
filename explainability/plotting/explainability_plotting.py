import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import librosa
import librosa.display
from scipy.ndimage import gaussian_filter1d
import os
from explainability.Gradient_based.saliency_map import interpolate_saliency
from explainability.plotting.utils import compute_log_spectrogram ,compute_formants
import base64
import io
from io import BytesIO


def set_time_ticks_ms(ax, total_duration, step_ms=1000, rotation=45):
    """
    Set x-axis ticks in milliseconds with labeled ticks every `step_ms`.
    Shows ticks every second but only labels every 3 seconds.

    Parameters:
    - ax: The matplotlib axis to apply the ticks to.
    - total_duration: Total duration of the audio in seconds.
    - step_ms: Step between ticks in milliseconds (default 1000 ms = 1 sec).
    - rotation: Label rotation angle (default 45 degrees).
    """
    # Generate ticks every second (1000 ms)
    time_ticks_ms = np.arange(0, int(total_duration * 1000) + step_ms, step_ms)
    ax.set_xticks(time_ticks_ms / 1000)  # Convert to seconds
    
    # Generate labels (only every 3 seconds)
    labels = [
        str(int(t / 1000)) if (int(t / 1000) % 3 == 0) else "" 
        for t in time_ticks_ms
    ]
    
    ax.set_xticklabels(labels, rotation=rotation, fontsize=8)
    ax.set_xlabel("Time (seconds)")
    
def plot_waveform_and_saliency(ax, total_duration, saliency_data):
    ax.plot(saliency_data['time'], saliency_data['waveform'], label="Waveform", alpha=0.7)
    ax.plot(saliency_data['time'], saliency_data['saliency'], label="Saliency", color="red", alpha=0.7)

    ax.set_title("Waveform + Saliency")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    set_time_ticks_ms(ax, total_duration)

def plot_colored_waveform(
    ax=None,
    total_duration=None,
    saliency_data=None,
    plot_method='threshold',
    threshold=0.5,
    cmap_name="Reds",
    min_saliency=0.1,
    return_base64=False,
    figsize = (10, 3),
    include_title=True
):
    """
    Plot a waveform where segments with saliency above a threshold are red,
    and others are gray. Optionally return base64 image for HTML embedding.
    """
    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig_created = True
    else:
        fig = ax.figure

    saliency = saliency_data['saliency']
    waveform = saliency_data['waveform']
    time = saliency_data['time']

    # Create line segments
    points = np.array([time, waveform]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if plot_method == "threshold":
        saliency_avg = (saliency[:-1] + saliency[1:]) / 2
        high_saliency_segments = segments[saliency_avg > threshold]
        low_saliency_segments = segments[saliency_avg <= threshold]
        lc_high = LineCollection(high_saliency_segments, colors='#ae0344', linewidth=1.0)
        lc_low = LineCollection(low_saliency_segments, colors='lightgray', linewidth=1.0)
        ax.add_collection(lc_low)
        ax.add_collection(lc_high)
    else:
        saliency_norm = np.where(saliency < min_saliency, min_saliency, saliency)
        saliency_avg = (saliency_norm[:-1] + saliency_norm[1:]) / 2
        cmap = plt.get_cmap(cmap_name)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), linewidth=1.0)
        lc.set_array(saliency_avg)
        ax.add_collection(lc)

   

    # Set limits and ticks
    margin = 0.02 * (np.max(waveform) - np.min(waveform))
    ax.set_ylim(np.min(waveform) - margin, np.max(waveform) + margin)
    set_time_ticks_ms(ax, total_duration)
    if include_title: ax.set_title("Waveform")
    ax.set_ylabel("Amplitude")

    if return_base64 and fig_created:
        from io import BytesIO
        import base64

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return image_base64

    return None


def plot_saliency_weighted_spectrogram(
    ax,
    audio_path,
    saliency_data,
    sr=16000,
    hop_length=512,
    merge_frame_duration=0.03,
    saliency_power=7,
    saliency_scale=20,
    saliency_min=0.01,
  ):
    """
    Plot a spectrogram with intensity modulated by saliency values (not SHAP).
    saliency_data: dict with 'saliency' and 'time' keys (same as your input format).
    """

    name = os.path.splitext(os.path.basename(audio_path))[0]

    # === Load audio and compute spectrogram ===
    audio, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=hop_length, power=2.0)
    duration = len(audio) / sr
    log_S = librosa.power_to_db(S, ref=np.max)

    # === Extract saliency and time ===
    saliency = saliency_data['saliency']
    sal_time = saliency_data['time']

    # === Interpolate saliency to match spectrogram frame times ===
    spec_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    saliency_interp = interpolate_saliency(saliency_data, spec_times)
    # saliency_interp = np.interp(spec_times, sal_time, saliency)

    # === Enhance and scale saliency ===
    norm = (saliency_interp - np.percentile(saliency_interp, 5)) / (
        np.percentile(saliency_interp, 95) - np.percentile(saliency_interp, 5) + 1e-6
    )
    norm = np.clip(norm, 0, 1)
    saliency_scaled = norm**saliency_power * saliency_scale + saliency_min

    # === Apply to spectrogram ===
    for i in range(S.shape[1]):
        S[:, i] *= saliency_scaled[i]

    modified_log_S = librosa.power_to_db(S, ref=np.max)

    # === Plot ===
    img = librosa.display.specshow(modified_log_S, sr=sr, x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_xlim(0, duration)
    ax.set_title(f"Saliency-Weighted Spectrogram ({name})")

    time_ticks_ms = np.arange(0, int(duration * 1000), 500)
    ax.set_xticks(time_ticks_ms / 1000)
    ax.set_xticklabels([str(int(t)) for t in time_ticks_ms], rotation=45)

def plot_saliency_weighted_spectrogram_blended(ax,total_duration, audio_path, saliency_data,
                                                highlight_cmap='viridis', base_cmap='gray_r',
                                                sr=16000, hop_length=512):

    _,spectrogram = compute_log_spectrogram(audio_path, sr, hop_length)
    frame_times = librosa.frames_to_time(np.arange(spectrogram.shape[1]), sr=sr, hop_length=hop_length)
    saliency_per_frame = interpolate_saliency(saliency_data, frame_times)

    # Normalize
    norm_spec = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-6)
    saliency = np.clip(saliency_per_frame, 0, 1)

    # Base spectrogram
    base_img = plt.get_cmap(base_cmap)(norm_spec)[..., :3]  # (freq, time, RGB)

    # Saliency overlay
    saliency_map = np.zeros_like(base_img)
    highlight_cm = plt.get_cmap(highlight_cmap)
    for j in range(spectrogram.shape[1]):
        sal_val = saliency[j]
        for i in range(spectrogram.shape[0]):
            saliency_map[i, j, :] = np.array(highlight_cm(norm_spec[i, j])[:3]) * sal_val  # Fixed

    # Blend saliency with base using alpha compositing
    # Per-pixel dynamic blending
    alpha = np.expand_dims(saliency, axis=0)  # shape: (1, time)
    alpha_map = np.tile(alpha, (spectrogram.shape[0], 1))  # shape: (freq, time)
    alpha_map = np.expand_dims(alpha_map, axis=-1)  # shape: (freq, time, 1)

    blended = (1 - alpha_map) * base_img + alpha_map * saliency_map


    ax.imshow(blended, aspect='auto', origin='lower',
              extent=[frame_times[0], frame_times[-1], 0, sr // 2])

    set_time_ticks_ms(ax, total_duration)

    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram with Saliency-Based Color Overlay")

def overlay_formants(ax, formants_data, audio_path):
    """
    Plots F0 and formant tracks. 
    """

    legend_handles = []

    try:
        ts_f0, ts_formants, values = compute_formants(audio_path, formants_data)
        colors = {"F0": 'red', "F1": 'cyan', "F2": 'white', "F3": '#FF8C00'}

        for name in formants_data:
            if name in values:
                line = ax.plot(
                    ts_f0 if name == "F0" else ts_formants,
                    values[name],
                    label=name,
                    color=colors[name],
                    linewidth=3 if name == "F0" else 2
                )[0]
                legend_handles.append(line)

    except Exception as e:
        print(f"Formant overlay failed: {e}")

    return legend_handles

def overlay_pauses(ax, pauses, max_y):
    """
    Draws pause rectangles on the given axis. Does not call ax.legend().
    Returns Patch objects for legend.
    """
    import matplotlib.patches as patches

    legend_handles = {}

    try:
        if not pauses:
            return []

        for start, end, *_, mark in pauses:
            linestyle = '-' if mark else '--'
            pause_rect = patches.Rectangle(
                (start, 0),
                end - start,
                max_y,
                linewidth=2,
                edgecolor="#ff8000",
                facecolor='none',
                linestyle=linestyle,
                alpha=0.8
            )
            ax.add_patch(pause_rect)

            key = "Informative Pause" if mark else "Natural Pause"
            if key not in legend_handles:
                legend_handles[key] = patches.Patch(
                    edgecolor="#ff8000",
                    facecolor='none',
                    linestyle=linestyle,
                    linewidth=2,
                    label=key
                )

        return list(legend_handles.values())

    except Exception as e:
        print(f"Pause overlay failed: {e}")
        return []



def plot_entropy(
    ax=None,
    total_duration=None,
    entropy_data=None,
    flat_segments=None,
    return_base64=False,
    figsize = (10, 3),
    include_title = True,
    legend_size=10
):
    """
    Plot spectral entropy over time and optionally highlight flat segments.
    Optionally return a base64 image for HTML embedding.
    """

    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig_created = True
    else:
        fig = ax.figure

    # Plot entropy curve
    ax.plot(entropy_data['times'], entropy_data['smoothed_entropy'], color='blue', label='Spectral Entropy')

    # Highlight flat segments
    if flat_segments:
        for i, (start, end) in enumerate(flat_segments):
            ax.axvspan(start, end, color='red', alpha=0.3, label='Flat Segment' if i == 0 else "")

    # Formatting

    
    if include_title: ax.set_title("Spectral Entropy")
    ax.set_ylabel("Normalized Entropy (bits)")
    set_time_ticks_ms(ax, total_duration)
    ax.legend(loc='upper right', prop={'size': legend_size})
    ax.grid(True, alpha=0.2)
    ax.set_xlim(left=0)

    # Return as base64 image if requested
    if return_base64 and fig_created:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return image_base64

    return None



def apply_blur(spectrogram, blur_mask, sigma=2):
    blurred = gaussian_filter1d(spectrogram, sigma=sigma, axis=1)
    result = spectrogram * (1 - blur_mask) + blurred * blur_mask
    return result

def compute_time_masks(shap_values, sr, merge_frame_duration, S, mode, overlay_negatives, blur_negatives,min_opacity=0.65):
    merged_shap_values = shap_values
    shap_norm = (merged_shap_values - np.min(merged_shap_values)) / (np.max(merged_shap_values) - np.min(merged_shap_values) + 1e-6)
    frame_duration = merge_frame_duration
    time_bins = np.linspace(0, len(shap_values) * frame_duration, S.shape[1])

    mask_overlay = np.zeros(S.shape[1])
    alpha_mask = np.ones(S.shape[1])
    blur_mask = np.zeros(S.shape[1])

    for i, value in enumerate(merged_shap_values):
        start_time = i * frame_duration
        end_time = start_time + frame_duration
        idx_start = np.searchsorted(time_bins, start_time)
        idx_end = np.searchsorted(time_bins, end_time)

        norm_value = shap_norm[i]

        if mode in [1, 2]:
            if value <= 0:
                if overlay_negatives:
                    mask_overlay[idx_start:idx_end] = 1
                if blur_negatives:
                    blur_mask[idx_start:idx_end] = 1
            elif mode == 2:

                alpha_mask[idx_start:idx_end] = min_opacity + (1 - min_opacity) * norm_value

        elif mode == 3:
            alpha_mask[idx_start:idx_end] = min_opacity + (1 - min_opacity) * norm_value

    return mask_overlay, alpha_mask, blur_mask

def plot_SHAP_highlighted_spectrogram(
        ax=None,
        total_duration=None,
        audio_path=None,
        shap_values=None,
        pauses=None,
        formants_data=None,
        label=None,
        visualization_mode=1,
        overlay_negatives=True,
        blur_negatives=False,
        colormap='viridis',
        overlay_color='#5f5f5f',
        sr=16000,
        hop_length=512,
        segment_length=5,
        overlap=0.2,
        merge_frame_duration=0.3,
        fade_alpha=0.3,
        blur_sigma=3,
        return_base64=False,
        figsize = (10, 4),
        include_title = True,
        legend_size=10
):
    """
    Plot a spectrogram with SHAP-based visual highlighting.
    Optionally returns a base64-encoded image for HTML embedding.
    """

    fig_created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig_created = True
    else:
        fig = ax.figure


    # Load and compute spectrogram
    audio, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=hop_length, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)

    if total_duration is None:
        total_duration = librosa.get_duration(y=audio, sr=sr)

    # Aggregate SHAP values
    shap_values_label = shap_values[0, :, :, label]
    merge_samples = int(merge_frame_duration * sr)

    merged_shap_values = []
    for segment in shap_values_label:
        trimmed = segment[: len(segment) // merge_samples * merge_samples]
        reshaped = trimmed.reshape(-1, merge_samples)
        merged_shap_values.append(reshaped.mean(axis=1))
    merged_shap_values = np.concatenate(merged_shap_values)

    # Compute masks
    mask_overlay, alpha_mask, blur_mask = compute_time_masks(
        merged_shap_values, sr, merge_frame_duration, S,
        mode=visualization_mode,
        overlay_negatives=overlay_negatives,
        blur_negatives=blur_negatives
    )

    if blur_negatives:
        log_S = apply_blur(log_S, blur_mask[np.newaxis, :], sigma=blur_sigma)

    # Time axis
    frame_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    # Plot spectrogram
    for i in range(S.shape[1]):
        ax.imshow(log_S[:, i:i + 1], aspect='auto', origin='lower',
                  extent=[frame_times[i], frame_times[min(i + 1, len(frame_times) - 1)], 0, sr // 2],
                  cmap=colormap, alpha=alpha_mask[i])

    # Overlay negatives
    if overlay_negatives:
        for i in range(S.shape[1]):
            if mask_overlay[i] > 0:
                ax.axvspan(frame_times[i], frame_times[min(i + 1, len(frame_times) - 1)],
                           ymin=0, ymax=1, color=overlay_color, alpha=fade_alpha)
                
    legend_handles = []

    if pauses:
        pause_handles = overlay_pauses(ax, pauses, sr // 2)
        legend_handles.extend(pause_handles)
    if formants_data:
        formant_handles = overlay_formants(ax, formants_data, audio_path)
        legend_handles.extend(formant_handles)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', prop={'size': legend_size})


    if include_title: ax.set_title(f"Spectrogram with SHAP Highlights")
    ax.set_ylabel("Frequency (Hz)")
    set_time_ticks_ms(ax, total_duration)

    # Return image if needed
    if return_base64 and fig_created:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return image_base64

    return None





def plot_SHAP_spectrogram(
    ax,
    total_duration,
    audio_path,
    shap_values,
    label,
    sr=16000,
    hop_length=512,
    segment_length=5,
    overlap=0.2,
    merge_frame_duration=0.3,
):


    name = os.path.splitext(os.path.basename(audio_path))[0]


    # Step 1: Load audio and compute spectrogram
    audio, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=hop_length, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)  # Convert to decibels

    # Step 2: Aggregate SHAP values for the specified label
    shap_values_label = shap_values[0, :, :, label]
    shap_values_label = np.maximum(shap_values_label, 0)

    segment_samples = int(segment_length * sr)
    hop_samples = int(segment_samples * (1 - overlap))

    # Merge SHAP values into larger frames
    merge_samples = int(merge_frame_duration * sr)
    merged_shap_values = []
    for segment in shap_values_label:
        reshaped_segment = segment[: len(segment) // merge_samples * merge_samples]
        reshaped_segment = reshaped_segment.reshape(-1, merge_samples)
        merged_shap_values.append(reshaped_segment.mean(axis=1))
    merged_shap_values = np.concatenate(merged_shap_values)


    # Normalize SHAP values for enhanced contrast
    merged_shap_values_normalized = (merged_shap_values - np.percentile(merged_shap_values, 5)) / (
        np.percentile(merged_shap_values, 95) - np.percentile(merged_shap_values, 5)
    )
    merged_shap_values_normalized = np.clip(merged_shap_values_normalized, 0, 1)

    # Apply nonlinear transformation for more intensity difference
    merged_shap_values_transformed = merged_shap_values_normalized**5
    merged_shap_values_transformed *= 800
    merged_shap_values_transformed += 1e-6

    # Step 3: Modify the spectrogram intensity
    audio_duration = len(audio) / sr
    merged_frame_times = np.arange(0, len(merged_shap_values)) * merge_frame_duration
    time_bins = np.linspace(0, audio_duration, S.shape[1])

    for i, t in enumerate(merged_frame_times):
        idx_start = np.searchsorted(time_bins, t)
        idx_end = np.searchsorted(time_bins, t + merge_frame_duration)
        if idx_start < len(S[0]) and idx_end < len(S[0]):
            S[:, idx_start:idx_end] *= merged_shap_values_transformed[i]

    # Step 5: Plot the spectrogram
    labels = {0: "Healthy", 1: "MCI", 2: "ADRD"}

    modified_log_S = librosa.power_to_db(S, ref=np.max)
    # img = librosa.display.specshow(modified_log_S, sr=sr, x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
    frame_times = librosa.frames_to_time(np.arange(modified_log_S.shape[1]), sr=sr, hop_length=512)
    img = ax.imshow(modified_log_S, aspect='auto', origin='lower',
                    extent=[frame_times[0], frame_times[-1], 0, sr // 2],
                    cmap='viridis')
    # Determine max_mel from the y_coords of the spectrogram
    max_mel = img.axes.yaxis.get_data_interval()[-1]  # Get the maximum y-axis value (mel frequency)

    ax.set_title(f"Spectrogram SHAP-weighted")
    set_time_ticks_ms(ax, total_duration)


def plot_spectrogram(
    ax,
    audio_path,
    sr=16000,
    total_duration=None,
    pauses =None,
    formants_data = None,
    hop_length=512,
    cmap='viridis',
    title="Spectrogram",
    legend_size = 10
  ):


    audio, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=hop_length, power=2.0)

    if total_duration is None:
        total_duration = librosa.get_duration(y=audio, sr=sr)

    log_S = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel", cmap=cmap, ax=ax)

    legend_handles = []

    if pauses:
        pause_handles = overlay_pauses(ax, pauses, sr // 2)
        legend_handles.extend(pause_handles)
    if formants_data:
        formant_handles = overlay_formants(ax, formants_data, audio_path)
        legend_handles.extend(formant_handles)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', prop={'size': legend_size})


    if title: ax.set_title(title)

    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{title}")
    set_time_ticks_ms(ax, total_duration)



# ===================== Main Plot Function =====================

def plot_all_audio_analysis(
    saliency_data,
    audio_path,
    sr,
    hop_length,
    sample_info,
    waveform_plot_method="gradient", #threshold
    SHAP_values=None,
    entropy_data=None,
    flat_segments=None,
    pauses=None,
    formants_data=None,
    include_axes=('waveform', 'colored_waveform','saliency_spectrogram', 'shap_spectrogram', 'plain_spectrogram', 'entropy'),
    overlay_options={'waveform': {'pauses': True},
                     'saliency_spectrogram': {'formants': True},
                     'shap_spectrogram': {'formants': True},
                     'plain_spectrogram': {'formants': True}},
    figsize=(18, 14),
    highlight_cmap='viridis',
    base_cmap='gray_r',
    save_path=None
  ):
    waveform = saliency_data['waveform']
    saliency = saliency_data['saliency']
    time = saliency_data['time']
    total_duration =  max(saliency_data['time'][-1], entropy_data['times'][-1])
    label = sample_info['label']

    height_ratios_map = {
    'waveform': 2,
    'colored_waveform': 2,
    'saliency_spectrogram':4.0,
    'shap_spectrogram': 4.0,
    'shap_spectrogram_1': 4.0,
    'shap_spectrogram_2': 4.0,
    'plain_spectrogram': 4.0,
    'entropy': 1.2
    }
    height_ratios = [height_ratios_map.get(name, 1.5) for name in include_axes]

    axis_plotters = {
        'waveform': lambda ax: plot_waveform_and_saliency(ax, total_duration, saliency_data),
        'colored_waveform': lambda ax: plot_colored_waveform(ax, total_duration, saliency_data,plot_method=waveform_plot_method ,threshold=0.5, cmap_name="Reds",min_saliency=0.1),
        'saliency_spectrogram':lambda ax: plot_saliency_weighted_spectrogram_blended(ax,total_duration, audio_path, saliency_data),
        'shap_spectrogram_1': lambda ax: plot_SHAP_highlighted_spectrogram(ax,total_duration, audio_path, shap_values=SHAP_values, label=label,visualization_mode=1),
        'shap_spectrogram_2': lambda ax: plot_SHAP_highlighted_spectrogram(ax,total_duration, audio_path, shap_values=SHAP_values, label=label,visualization_mode=2),
        'shap_spectrogram': lambda ax: plot_SHAP_spectrogram(ax,total_duration, audio_path, shap_values=SHAP_values, label=label),
        'plain_spectrogram': lambda ax: plot_spectrogram(ax,total_duration, audio_path),
        'entropy': lambda ax: plot_entropy(ax,total_duration, entropy_data, flat_segments)
    }

    num_axes = len(include_axes)
    fig, axes = plt.subplots(
        num_axes, 1,
        figsize=figsize,
        sharex=False,
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.3},
    )
    if num_axes == 1:
        axes = [axes]  # Ensure iterable

    fig.suptitle(
        f"Analysis for sample: {sample_info['uid']} ; class: {sample_info['class']}",
        fontsize=16, y=0.92
    )

    for idx, axis_name in enumerate(include_axes):
        ax = axes[idx]
        ax.set_xlim(0, total_duration)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Plot the main data
        axis_plotters[axis_name](ax)

        # Optional overlays
        overlay_opts = overlay_options.get(axis_name, {})

        legend_handles = []

        if overlay_opts.get('pauses') and pauses:
            if 'spectrogram' in axis_name:
                pause_handles = overlay_pauses(ax, pauses, sr // 2)
            elif 'waveform' in axis_name:
                pause_handles = overlay_pauses(ax, pauses, max(np.max(waveform), np.max(saliency)))
            else:
                pause_handles = []
            legend_handles.extend(pause_handles)

        if 'spectrogram' in axis_name and overlay_opts.get('formants') and formants_data:
            formant_handles = overlay_formants(ax, formants_data, audio_path)
            legend_handles.extend(formant_handles)

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right')


        if idx == num_axes - 1:
            ax.set_xlabel("Time (s)", fontsize=10)


    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()
