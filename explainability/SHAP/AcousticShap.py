import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import shap
import librosa
import parselmouth

from scipy.signal import welch

from typing import Optional, Tuple, Union, Any,List,Dict

import librosa.display
from scipy.signal import butter, filtfilt, welch

DEFAULT_CUTOFF_FREQ = 8000


class AcousticShap():
    
    def __init__(self,model):
        self.model = model
        self.default_sr = 16000
        self.default_cutoff = DEFAULT_CUTOFF_FREQ
        
    
    def get_speech_shap_results(
        self,
        audio_path,
        demography_info,
        config,
        frame_duration=0.3,
        formants_to_plot=["F0", "F3"],
        segment_length=5,
        overlap=0.2,
        target_sr=16000,
        baseline_type='zeros'
    ):
        """
        Calculates SHAP values for the given audio file, creates a figure with a spectrogram
        and frequency Shannon entropy subplots, saves the figure to fig_save_path, and returns the figure.
        """
        audio_path = str(audio_path)
        audio_label = self.model.inference(audio_path, demography_info, config)[0]

        shap_results = self.calculate_speech_shap_values(
            audio_path,
            segment_length=segment_length,
            overlap=overlap,
            target_sr=target_sr,
            baseline_type=baseline_type,
        )
        shap_values = shap_results["shap_values"]
        shap_values_aggregated = shap_results["shap_values_aggregated"]
        predictions = shap_results["predictions"]

        # Create the figure and grid
        fig = plt.figure(figsize=(20, 5.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.5])

        # Spectrogram subplot
        ax0 = plt.subplot(gs[0])
        _ = self.visualize_shap_spectrogram(
            audio_path,
            shap_values,
            audio_label,
            sr=target_sr,
            segment_length=segment_length,
            overlap=overlap,
            merge_frame_duration=frame_duration,
            formants_to_plot=formants_to_plot,
            fig_save_path=None,
            ax=ax0
        )

        # Frequency Shannon Entropy subplot
        ax1 = plt.subplot(gs[1])
        _ = self.frequency_shannon_entropy(
            audio_path,
            ax=ax1,
            smooth_window=50
        )

        plt.tight_layout()
        # Ensure the directory exists and save the figure
        fig_save_path = f"speech_shap_{os.path.basename(audio_path)}.png"
        plt.savefig(fig_save_path, dpi=600, bbox_inches="tight", transparent=True)
        return fig_save_path
    
    def calculate_speech_shap_values(
        self,
        audio_path,
        segment_length=5,
        overlap=0.2,
        target_sr=16000,
        baseline_type='zeros'
    ):
        result = self.model.speech_only_inference(
            audio_path,
            segment_length=segment_length,
            overlap=overlap,
            target_sr=target_sr,
            device=self.model.device
        )

        segments_tensor = torch.tensor(result["segments_tensor"]).to(self.model.device)  # Input tensor for SHAP
        predictions = result["predictions"]

        if baseline_type == 'zeros':
            baseline_data = torch.zeros_like(segments_tensor)  # Zero baseline
        elif baseline_type == 'mean':
            baseline_data = torch.mean(segments_tensor, dim=0, keepdim=True).repeat(
                segments_tensor.size(0), 1, 1
            )  # Mean baseline

        baseline_data = baseline_data.unsqueeze(0) if baseline_data.dim() == 2 else baseline_data
        segments_tensor = segments_tensor.unsqueeze(0) if segments_tensor.dim() == 2 else segments_tensor

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
        
            def forward(self, x):
                # Instead of calling self.model.forward(x)
                return self.model.speech_only_forward(x)
        
        explainer = shap.DeepExplainer(ModelWrapper(self.model), baseline_data)

        shap_values = explainer.shap_values(segments_tensor, check_additivity=False)  # Disable additivity check

        shap_values_aggregated = [shap_val.sum(axis=-1) for shap_val in shap_values]

        return {
            "shap_values": shap_values,
            "shap_values_aggregated": shap_values_aggregated,
            "segments_tensor": segments_tensor.cpu().numpy(),
            "predictions": predictions
        }
    
    def get_speech_spectrogram(
        self,
        audio_path: str,
        demography_info: Any,
        config: dict,
        *,
        spectrogram_type: str = "original",
        include_entropy: bool = False,
        entropy_kwargs: Optional[Dict] = None,
        formants_to_plot: Optional[List[str]] = None,
        pauses: Optional[List[Tuple[float, float]]] = None,
        sr: int = None,
        segment_length: float = 5,
        overlap: float = 0.2,
        frame_duration: float = 0.3,
        baseline_type: str = 'zeros',
        fig_save_path: Optional[str] = None,
        plot: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Enhanced spectrogram analysis with spectral entropy and formant tracking support.
        
        Args:
            audio_path: Path to audio file
            demography_info: Demographic information for model
            config: Model configuration dictionary
            spectrogram_type: Type of spectrogram ("original" or "shap")
            include_entropy: Whether to include spectral entropy analysis
            entropy_kwargs: Parameters for plotting entropy
            formants_to_plot: List of formant names to plot (e.g., ['F1', 'F2'])
            pauses: List of pause intervals as (start, end) tuples
            sr: Target sampling rate (uses default if None)
            segment_length: SHAP segment length in seconds
            overlap: SHAP segment overlap ratio
            frame_duration: SHAP merge frame duration in seconds
            baseline_type: SHAP baseline type
            fig_save_path: Path to save output figure
            plot: Whether to display the plot
            
        Returns:
            Returns vary based on requested analyses:
            - Spectrogram only (default)
            - (spectrogram, entropy, flat_segments) if include_entropy=True
            - (spectrogram, formant_values) if formants_to_plot is not empty
            - (spectrogram, entropy, flat_segments, formant_values) if both are requested
            
        Raises:
            ValueError: For invalid spectrogram_type
        """
        sr = sr or self.default_sr
        
        # Validate inputs
        if spectrogram_type not in ["original", "shap"]:
            raise ValueError("spectrogram_type must be 'original' or 'shap'")
            
        if formants_to_plot is None:
            formants_to_plot = []

        # Generate appropriate spectrogram and formants
        formant_values, spectrogram = self._generate_spectrogram(
            spec_type=spectrogram_type,
            audio_path=audio_path,
            demography_info=demography_info,
            config=config,
            formants=formants_to_plot,
            pauses=pauses,
            sr=sr,
            segment_length=segment_length,
            overlap=overlap,
            frame_duration=frame_duration,
            baseline_type=baseline_type,
            plot=plot
        )

        # Initialize return values
        return_values = [spectrogram]
        
        # Calculate and plot entropy if requested
        if include_entropy:
            entropy, flat_segments = self.plot_entropy_analysis(
                audio_path,
                sr=sr,
                **entropy_kwargs if entropy_kwargs else {}
            )
            return_values.extend([entropy, flat_segments])
        
        # Add formant values if requested
        if formants_to_plot:
            return_values.append(formant_values)
        
        # Return appropriate tuple based on what was calculated
        return tuple(return_values) if len(return_values) > 1 else return_values[0]
    
    def _generate_spectrogram(
        self,
        spec_type: str,
        audio_path: str,
        demography_info: Any,
        config: dict,
        formants: List[str],
        pauses: List[Tuple[float, float]],
        sr: int,
        segment_length: float,
        overlap: float,
        frame_duration: float,
        baseline_type: str,
        plot: bool= True
    ) -> np.ndarray:
        """Internal method to generate appropriate spectrogram."""
        fig, ax = plt.subplots(figsize=(20, 4))
        if spec_type == "original":
            return self.visualize_original_spectrogram(
                ax=ax,
                audio_path=audio_path,
                sr=sr,
                formants_to_plot=formants,
                pauses=pauses,
                fig_save_path=None,
                plot=plot
            )
        else:
        
            audio_label = self.model.inference(audio_path, demography_info, config)[0]
            shap_values = self.calculate_speech_shap_values(
                audio_path,
                segment_length=segment_length,
                overlap=overlap,
                target_sr=sr,
                baseline_type=baseline_type,
            )["shap_values"]
            
            return self.visualize_shap_spectrogram(
                ax=ax,
                audio_path=audio_path,
                shap_values=shap_values,
                label=audio_label,
                sr=sr,
                segment_length=segment_length,
                overlap=overlap,
                merge_frame_duration=frame_duration,
                formants_to_plot=formants,
                pauses=pauses,
                fig_save_path=None,
                plot=plot
            )


    def _low_pass_filter(
        self,
        signal: np.ndarray,
        sr: int,
        cutoff: int,
        order: int = 6
    ) -> np.ndarray:
        """Apply low-pass Butterworth filter."""
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def _shannon_entropy(self, stft: np.ndarray) -> np.ndarray:
        """Calculate Shannon entropy from STFT magnitudes."""
        ps = np.abs(stft)**2
        ps_norm = ps / np.sum(ps, axis=0, keepdims=True)
        ps_norm[ps_norm == 0] = 1e-12  # Avoid log(0)
        return -np.sum(ps_norm * np.log2(ps_norm), axis=0)

    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth signal using moving average."""
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    
    def detect_flat_segments(
        self,
        entropy: np.ndarray,
        times: np.ndarray,
        window_size: int = 10,
        std_threshold: float = 0.1,
        merge_gap: float = 0.3,
        segment_min_duration: float = 1,
        final_min_duration: float = 2
    ) -> List[Tuple[float, float]]:
        """
        Detect flat segments in entropy signal with advanced merging.
        
        Args:
            entropy: Array of entropy values
            times: Corresponding time points
            window_size: Rolling window size for std calculation
            std_threshold: Maximum std deviation to consider flat
            merge_gap: Maximum gap between segments to merge (seconds)
            segment_min_duration: Minimum duration for initial segments (seconds)
            final_min_duration: Minimum duration for final segments (seconds)
            
        Returns:
            List of (start, end) tuples for flat segments
        """
        # Calculate rolling standard deviation
        half_window = window_size // 2
        smooth_std = np.array([
            np.std(entropy[max(0, i - half_window):min(len(entropy), i + half_window)])
            for i in range(len(entropy))
        ])
        
        # Initial flat segment detection
        flat_mask = smooth_std < std_threshold
        segments = self._find_initial_segments(flat_mask, times, segment_min_duration)
        
        # Merge nearby segments
        merged_segments = self._merge_segments(segments, merge_gap)
        
        # Filter by final duration
        return [
            (start, end) for start, end in merged_segments
            if end - start >= final_min_duration
        ]

    def _find_initial_segments(
        self,
        flat_mask: np.ndarray,
        times: np.ndarray,
        min_duration: float
    ) -> List[Tuple[float, float]]:
        """Identify initial flat segments meeting duration requirement."""
        segments = []
        start_idx = None
        
        for i in range(1, len(flat_mask)):
            if flat_mask[i] and not flat_mask[i - 1]:
                start_idx = i
            elif not flat_mask[i] and flat_mask[i - 1] and start_idx is not None:
                if times[i] - times[start_idx] >= min_duration:
                    segments.append((times[start_idx], times[i]))
        
        return segments

    def _merge_segments(
        self,
        segments: List[Tuple[float, float]],
        max_gap: float
    ) -> List[Tuple[float, float]]:
        """Merge nearby segments with small gaps between them."""
        if not segments:
            return []
            
        merged = [list(segments[0])]
        
        for current_start, current_end in segments[1:]:
            last_start, last_end = merged[-1]
            
            if current_start - last_end <= max_gap:
                merged[-1][1] = current_end  # Extend previous segment
            else:
                merged.append([current_start, current_end])
                
        return [tuple(seg) for seg in merged]

    def plot_entropy_analysis(
        self,
        audio_path: str,
        sr: Optional[int] = None,
        cutoff: Optional[int] = None,
        window_size: int = 15,
        min_duration: float = 5,
        segments_std_threshold: float = 0.4,
        segments_merge_gap: float = 0.5,
        figsize: Tuple[int, int] = (20, 2),
        dpi: int = 200
    ) -> List[Tuple[float, float]]:
        """
        Complete entropy analysis with visualization.
        
        Args:
            audio_path: Path to audio file
            sr: Target sampling rate
            cutoff: Low-pass cutoff frequency
            window_size: Smoothing window size
            min_duration: Minimum flat segment duration
            segments_std_threshold: Maximum standard deviation threshold for 
                                   considering a segment as "flat" (lower values 
                                   detect more stable segments)
            segments_merge_gap: Maximum time gap in seconds between adjacent flat 
                              segments to allow merging
            figsize: Figure dimensions
            dpi: Figure resolution
            
        Returns:
            Value of the calculated entropy
            List of detected flat segments as (start, end) tuples
        """
        sr = sr or self.default_sr
        cutoff = cutoff or self.default_cutoff
        
        # Load and process audio
        signal, orig_sr = librosa.load(audio_path, sr=None)
        filtered_signal = self._low_pass_filter(signal, orig_sr, cutoff)
        resampled_signal = librosa.resample(filtered_signal, orig_sr=orig_sr, target_sr=sr)
        
        # Calculate entropy
        stft = librosa.stft(resampled_signal, n_fft=1024, hop_length=512)
        entropy = self._shannon_entropy(stft)
        times = librosa.frames_to_time(np.arange(len(entropy)), sr=sr, hop_length=512)
        smoothed_entropy = self._smooth_signal(entropy, window_size)
        
        # Detect flat segments
        flat_segments = self.detect_flat_segments(
            smoothed_entropy,
            times[:len(smoothed_entropy)],
            std_threshold=segments_std_threshold,
            merge_gap=segments_merge_gap,
            final_min_duration=min_duration
        )
        
        # Create visualization
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(times[:len(smoothed_entropy)], smoothed_entropy, 
                label='Smoothed Shannon Entropy', color='blue')
        
        for start, end in flat_segments:
            plt.axvspan(start, end, color='red', alpha=0.3, 
                       label='Flat Segment' if start == flat_segments[0][0] else "")
        
        plt.xticks(np.arange(0, np.max(times), step=1))
        plt.grid(axis='x')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Entropy (bits)')
        plt.title('Spectral Entropy Analysis')
        plt.tight_layout()
        plt.show()
        
        print("Detected flat segments (start, end in seconds):")
        for seg in flat_segments:
            print(f"{seg[0]:.2f} - {seg[1]:.2f}")
        
        return entropy, flat_segments

    
    def moving_average(self, data, window_size=5):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def visualize_original_spectrogram(
        self,
        audio_path,
        sr=16000,
        formants_to_plot=None,
        pauses=None,
        fig_save_path=None,
        ax=None,
        plot=False
    ):
        """
        Visualize and return the original spectrogram with formants and pauses.
        Matches the style of visualize_shap_spectrogram() exactly except for SHAP modifications.
        
        Args:
            audio_path (str): Path to audio file
            sr (int): Sample rate
            formants_to_plot (list): Formants to overlay (e.g., ["F0", "F1"])
            pauses (list): List of pause intervals to mark
            fig_save_path (str): Path to save figure
            ax (matplotlib.axes): Existing axis to plot on
            plot (bool): Whether to display the plot
            
        Returns:
            np.ndarray: The original log-power mel spectrogram in dB
        """
        # Step 1: Load audio and compute spectrogram
        audio, _ = librosa.load(audio_path, sr=sr)
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, power=2.0)
        log_S = librosa.power_to_db(S, ref=np.max)
        audio_duration = len(audio) / sr

        # Step 2: Create figure (same style as SHAP version)
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 4))

        # Plot original spectrogram (same colormap as SHAP version)
        img = librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel", 
                                     cmap="viridis", ax=ax)
        
        # Get max mel frequency for pause plotting
        max_mel = img.axes.yaxis.get_data_interval()[-1]

        # Step 3: Plot pauses (identical to SHAP version)
        if pauses:
            ax.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], color="yellow", linestyle="-", linewidth=2, label="Informative Pause")
            ax.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], color="yellow", linestyle="--", linewidth=2, label="Natural Pause")
       
            for start, end, _, _, _, _, mark in pauses:
                linestyle = "-" if mark else "--"
                ax.plot([start, start, end, end, start], 
                        [0, max_mel, max_mel, 0, 0], 
                        color="yellow", 
                        linestyle=linestyle, 
                        linewidth=2)

        # Step 4: Axis formatting (identical to SHAP version)
        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Frequency (Hz)", fontsize=16)
        ax.set_xlim(0, audio_duration)
        
        # X-axis ticks in milliseconds (same as SHAP version)
        time_ticks_ms = np.arange(0, audio_duration * 1000, 500)  # Every 500 ms
        time_ticks_seconds = time_ticks_ms / 1000
        ax.set_xticks(time_ticks_seconds)
        ax.set_xticklabels([f"{int(t)}" for t in time_ticks_ms], rotation=45)

        # Step 5: Formant plotting (identical to SHAP version)
        formant_values = {}
        if formants_to_plot:
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            time_stamps = pitch.ts()
            f0_values = pitch.selected_array["frequency"]
            f0_values[f0_values == 0] = np.nan
            formant = sound.to_formant_burg(time_step=0.1)
            times = np.arange(0, audio_duration, 0.01)
            formant_values = {"F0": f0_values, "F1": [], "F2": [], "F3": []}
            for t in times:
                formant_values["F1"].append(formant.get_value_at_time(1, t))
                formant_values["F2"].append(formant.get_value_at_time(2, t))
                formant_values["F3"].append(formant.get_value_at_time(3, t))

            formant_colors = {"F0": 'red', "F1": 'cyan', "F2": 'white', "F3": '#FF8C00'}
            for formant in formants_to_plot:
                if formant in formant_values:
                    ax.plot(
                        times if formant != "F0" else time_stamps,
                        formant_values[formant],
                        label=formant,
                        linewidth=3 if formant == "F0" else 2,
                        color=formant_colors[formant]
                    )
            ax.legend(loc='upper right')

        # Step 6: Save/show (same as SHAP version)
        if fig_save_path:
            plt.savefig(fig_save_path, dpi=600, bbox_inches="tight")
        if plot:
            plt.show()
            
        return formant_values , log_S
    
    def visualize_shap_spectrogram(
        self,
        audio_path,
        shap_values,
        label,
        sr=16000,
        segment_length=5,
        overlap=0.2,
        merge_frame_duration=0.3,
        formants_to_plot=None,
        fig_save_path=None,
        pauses = None,
        ax=None,
        plot=False
    ):
        """
        Visualize the spectrogram with intensity modified by SHAP values, with optional formant plotting.

        Args:
            audio_path (str): Path to the audio file.
            shap_values (np.ndarray): SHAP values of shape (1, num_segments, seq_length, num_labels).
            label (int): The target label for visualization (0, 1, or 2).
            sr (int): Sampling rate of the audio file.
            segment_length (float): Length of each segment in seconds.
            overlap (float): Overlap ratio between segments.
            merge_frame_duration (float): Duration of merged frames in seconds.
            formants_to_plot (list): List of formants to plot (e.g., ["F0", "F1", "F2", "F3"]).
            fig_save_path (str, optional): Path to save the figure.
            pauses (list): List of pauses to plot. 
            ax (matplotlib.axes.Axes, optional): Axis to plot on for subplots. If None, creates a new plot.
            plot (bool): Whether to display the plot. Default is False. 

        Returns:
            None: Displays or saves the spectrogram.
        """
        name = os.path.splitext(os.path.basename(audio_path))[0]

        # Step 1: Load audio and compute spectrogram
        audio, _ = librosa.load(audio_path, sr=sr)
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, power=2.0)
        log_S = librosa.power_to_db(S, ref=np.max)  # Convert to decibels

        # Step 2: Aggregate SHAP values for the specified label
        shap_values_label = shap_values[0, :, :, label]
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
        merged_shap_values_transformed *= 10
        merged_shap_values_transformed += 0.01

        # Step 3: Modify the spectrogram intensity
        audio_duration = len(audio) / sr
        merged_frame_times = np.arange(0, len(merged_shap_values)) * merge_frame_duration
        time_bins = np.linspace(0, audio_duration, S.shape[1])

        for i, t in enumerate(merged_frame_times):
            idx_start = np.searchsorted(time_bins, t)
            idx_end = np.searchsorted(time_bins, t + merge_frame_duration)
            if idx_start < len(S[0]) and idx_end < len(S[0]):
                S[:, idx_start:idx_end] *= merged_shap_values_transformed[i]

        # Step 4: Extract formants if formants_to_plot is specified
        formant_values = {}
        if formants_to_plot:
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            time_stamps = pitch.ts()
            f0_values = pitch.selected_array["frequency"]
            f0_values[f0_values == 0] = np.nan
            formant = sound.to_formant_burg(time_step=0.1)
            times = np.arange(0, audio_duration, 0.01)
            formant_values = {"F0": f0_values, "F1": [], "F2": [], "F3": []}
            for t in times:
                formant_values["F1"].append(formant.get_value_at_time(1, t))
                formant_values["F2"].append(formant.get_value_at_time(2, t))
                formant_values["F3"].append(formant.get_value_at_time(3, t))

        # Step 5: Plot the spectrogram
        labels = {0: "Healthy", 1: "MCI", 2: "ADRD"}
        modified_log_S = librosa.power_to_db(S, ref=np.max)

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 4))

        img = librosa.display.specshow(modified_log_S, sr=sr, x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
        # ax.set_title(f"UID: {name}, Spectrogram with SHAP-Adjusted Intensity (Label: {labels[label]})")

        # Determine max_mel from the y_coords of the spectrogram
        max_mel = img.axes.yaxis.get_data_interval()[-1]  # Get the maximum y-axis value (mel frequency)

        # Plot pauses and create legend entries
        if pauses:
            # # Create proxy artists for the legend
            ax.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], color="yellow", linestyle="-", linewidth=2, label="Informative Pause")
            ax.plot([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], color="yellow", linestyle="--", linewidth=2, label="Natural Pause")
       
            for start, end, _, _, _, _, mark in pauses:
                # Use solid yellow for pauses with a mark, dashed yellow otherwise
                linestyle = "-" if mark else "--"
                ax.plot([start, start, end, end, start], [0, max_mel, max_mel, 0, 0], color="yellow", linestyle=linestyle, linewidth=2)

        ax.set_xlabel("Time (ms)", fontsize=16)  # Add x-axis label
        ax.set_ylabel("Frequency (Hz)", fontsize=16)
        ax.set_xlim(0, audio_duration)

        # Define x-axis ticks in milliseconds
        time_ticks_ms = np.arange(0, audio_duration * 1000, 500)  # Every 500 ms
        time_ticks_seconds = time_ticks_ms / 1000  # Convert ms to seconds for plotting
        ax.set_xticks(time_ticks_seconds)
        ax.set_xticklabels([f"{int(t)}" for t in time_ticks_ms], rotation=45)  # Display ticks in ms

        formant_colors = {"F0": 'red', "F1": 'cyan', "F2": 'white', "F3": '#FF8C00'}
        if formants_to_plot:
            for formant in formants_to_plot:
                if formant in formant_values:
                    ax.plot(
                        times if formant != "F0" else time_stamps,
                        formant_values[formant],
                        label=formant,
                        linewidth=3 if formant == "F0" else 2,
                        color=formant_colors[formant]
                    )
            ax.legend(loc='upper right')

        # plt.colorbar(img, ax=ax, format="%+2.0f dB")
        if fig_save_path:
            # folder_path = os.path.dirname(fig_save_path)
            # os.makedirs(folder_path, exist_ok=True)
            plt.savefig(fig_save_path, dpi=600, bbox_inches="tight")

        if plot: 
            plt.show()
            
        return formant_values, modified_log_S
