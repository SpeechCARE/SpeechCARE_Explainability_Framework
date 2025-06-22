import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import parselmouth
import librosa

def plot_multimodal_analysis(audio_path, saliency_data, spectrogram, grad_norm, entropy_data, formants_data,
                              pauses=None, flat_segments=None, 
                              sample_rate=16000, hop_length=512, cmap='hot', alpha=0.5):
    """
    Combined visualization of waveform saliency, spectrogram gradients, and entropy.
    """

    # Create figure with 3 vertically stacked subplots
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.5], figure=fig)

    # Shared time axis parameters
    total_duration = max(saliency_data['time'][-1], entropy_data['times'][-1])
    time_axis = np.linspace(0, total_duration, spectrogram.shape[1])

    # ================== 1. Waveform and Saliency ==================
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(saliency_data['time'], saliency_data['waveform'], label="Waveform", alpha=0.7, color='lightgray')
    ax1.plot(saliency_data['time'], saliency_data['saliency'], label="Saliency", color="red", alpha=0.7)

    if pauses:
        ax1.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], color="#043927", linestyle="-", linewidth=2, label="Informative Pause")
        ax1.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], color="#043927", linestyle="--", linewidth=2, label="Natural Pause")
        max_val = max(np.max(saliency_data['waveform']), np.max(saliency_data['saliency']))
        for start, end, *_, mark in pauses:
            linestyle = "-" if mark else "--"
            ax1.plot([start, start, end, end, start],
                     [0, max_val, max_val, 0, 0],
                     color="#043927", linestyle=linestyle, linewidth=2, alpha=0.8)
            ax1.fill_between([start, end], [0, 0], [max_val, max_val], color="#043927", alpha=0.1)

    ax1.set_xlim(0, total_duration)
    ax1.set_ylabel("Amplitude / Saliency")
    ax1.set_title(f"Waveform and Saliency (Class: {saliency_data['class_name']})")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ================== 2. Spectrogram Gradients ==================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Convert bin indices to actual frequencies (Hz)
    freq_axis_hz = librosa.fft_frequencies(sr=sample_rate, n_fft=(spectrogram.shape[0] - 1) * 2)

    spec_im = ax2.imshow(spectrogram, aspect='auto', origin='lower',
                        extent=[0, total_duration, freq_axis_hz[0], freq_axis_hz[-1]])

    # Upsample grad_norm to match spectrogram shape if needed
    grad_im = ax2.imshow(grad_norm[np.newaxis, :], aspect='auto', origin='lower',
                        extent=[0, total_duration, freq_axis_hz[0], freq_axis_hz[-1]],
                        cmap=cmap, alpha=alpha, interpolation='bilinear')

    fig.colorbar(spec_im, ax=ax2, label='Magnitude', pad=0.01, shrink=0.8)
    fig.colorbar(grad_im, ax=ax2, label='Gradient Importance', pad=0.01, shrink=0.8)


    ax2.set_ylabel("Frequency Bin")
    ax2.set_title(f"Spectrogram with Gradient Importance (Class: {saliency_data['class_name']})")

    # ========== Formants Plotting (optional) ==========
    if formants_data:
        try:
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            time_stamps = pitch.ts()
            f0_values = pitch.selected_array["frequency"]
            f0_values[f0_values == 0] = np.nan

            audio_duration = sound.get_total_duration()
            times = np.arange(0, audio_duration, 0.01)

            formant = sound.to_formant_burg(time_step=0.1)
            formant_values = {"F0": f0_values, "F1": [], "F2": [], "F3": []}

            for t in times:
                formant_values["F1"].append(formant.get_value_at_time(1, t))
                formant_values["F2"].append(formant.get_value_at_time(2, t))
                formant_values["F3"].append(formant.get_value_at_time(3, t))

            formant_colors = {"F0": 'red', "F1": 'cyan', "F2": 'white', "F3": '#FF8C00'}
            # formants_to = ["F0", "F1", "F2", "F3"]

            for formant_name in formants_data:
                if formant_name in formant_values:
                    ax2.plot(
                        time_stamps if formant_name == "F0" else times,
                        formant_values[formant_name],
                        label=formant_name,
                        linewidth=3 if formant_name == "F0" else 2,
                        color=formant_colors[formant_name]
                    )
            ax2.legend(loc='upper right')

        except Exception as e:
            print(f"Formant extraction failed: {e}")

    # ================== 3. Entropy Analysis ==================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(entropy_data['times'], entropy_data['smoothed_entropy'],
             label='Spectral Entropy', color='blue')

    if flat_segments:
        for start, end in flat_segments:
            ax3.axvspan(start, end, color='red', alpha=0.3,
                        label='Flat Segment' if start == flat_segments[0][0] else "")

    if pauses:
        for start, end, *_, mark in pauses:
            ax3.axvspan(start, end, color='yellow', alpha=0.1)

    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Entropy (bits)")
    ax3.set_title("Spectral Entropy Analysis")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    plt.show()
