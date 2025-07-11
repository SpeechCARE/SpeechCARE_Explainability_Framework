import numpy as np
from scipy.signal import butter, filtfilt
from typing import List, Tuple, Optional, Dict, Any
import librosa
import matplotlib.pyplot as plt
from explainability.plotting.explainability_plotting import plot_entropy


class EntropyAnalyzer:
    """
    A class for analyzing entropy in audio signals to detect stable/flat segments.

    Attributes:
        DEFAULT_CUTOFF_FREQ (int): Default cutoff frequency for low-pass filter
        default_sr (int): Default sampling rate
        default_cutoff (int): Default cutoff frequency
    """

    DEFAULT_CUTOFF_FREQ = 8000
    default_sr = 16000
    default_cutoff = DEFAULT_CUTOFF_FREQ

    def __init__(self):
        """Initialize the EntropyAnalyzer with default parameters."""
        pass


    def _low_pass_filter(self,signal, sr, cutoff=8000, order=6):
      """ Apply a low-pass Butterworth filter before resampling to prevent aliasing. """
      nyquist = sr / 2
      normal_cutoff = cutoff / nyquist
      b, a = butter(order, normal_cutoff, btype='low', analog=False)
      return filtfilt(b, a, signal)

    def _shannon_entropy(self,signal):
        """ Compute Shannon entropy from STFT magnitude. """
        ps = np.abs(signal)**2
        ps_norm = ps / np.sum(ps, axis=0, keepdims=True)
        ps_norm[ps_norm == 0] = 1e-12  # Avoid log issues
        return -np.sum(ps_norm * np.log2(ps_norm), axis=0)

    def _smooth_signal(self,signal, window_size=5):
        """ Smooth the signal using a moving average window. """
        return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')


    def _find_initial_segments(
        self,
        flat_mask: np.ndarray,
        times: np.ndarray,
        min_duration: float
    ) -> List[Tuple[float, float]]:
        """
        Identify initial flat segments meeting duration requirement.

        Args:
            flat_mask: Boolean array indicating flat regions
            times: Corresponding time points
            min_duration: Minimum duration for segments

        Returns:
            List of (start, end) time tuples
        """
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
        """
        Merge nearby segments with small gaps between them.

        Args:
            segments: List of (start, end) time tuples
            max_gap: Maximum gap between segments to allow merging

        Returns:
            List of merged (start, end) time tuples
        """
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


    def detect_flat_segments(self,entropy, times, final_min_duration, merge_gap, std_threshold,  ff_size):
        """
        Detects and merges relatively flat entropy segments.
        - ff_size: Number of frames to consider for measuring fluctuation
        - std_threshold: Max allowed std deviation to classify as "flat"
        - merge_gap: Max allowed gap (in seconds) to merge consecutive flat segments
        - min_duration: Minimum duration (in seconds) for consecutive flat segments to be considered
        """

        smooth_std = np.array([np.std(entropy[max(0, i - ff_size//2):i + ff_size//2])
                              for i in range(len(entropy))])


        flat_mask = smooth_std < std_threshold  # Identify flat areas
        segments = []
        start_idx = None

        for i in range(len(flat_mask)):
            if flat_mask[i]:
                if start_idx is None:
                    start_idx = i  # Start of a new flat segment
            else:
                if start_idx is not None:
                    end_idx = i
                    segments.append((times[start_idx], times[end_idx]))
                    start_idx = None

        # Handle case where flat segment continues till the end
        if start_idx is not None:
            segments.append((times[start_idx], times[-1]))

        # Merge consecutive segments if gap is small
        merged_segments = []
        for seg in segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_start, last_end = merged_segments[-1]
                current_start, current_end = seg
                if current_start - last_end <= merge_gap:
                    # Merge with previous segment
                    merged_segments[-1] = (last_start, current_end)
                else:
                    merged_segments.append(seg)


        final_segments = []
        for segment in merged_segments:
            start = segment[0]
            end = segment[1]
            if end - start >= final_min_duration:
                final_segments.append(segment)

        return final_segments


    def analyze(
        self,
        audio_path: str,
        sr: Optional[int] = None,
        cutoff: Optional[int] = None,
        window_size: int = 15,
        min_duration: float = 5,
        segments_std_threshold: float = 0.4,
        segments_merge_gap: float = 0.5,
        flucturation_frame_size: int =17,
        figsize: Tuple[int, int] = (10, 4),
        legend_size:int=10,
        dpi: int = 200,
        visualize: bool = False,
        filter_signal:bool = False,
        return_base64=False,
        ax=None,
        fig_save_path=None,
    ) -> Dict[str, Any]:

        sr = sr or self.default_sr
        cutoff = cutoff or self.default_cutoff

        # Load and process audio
        signal, orig_sr = librosa.load(audio_path, sr=None)

        if filter_signal:
            filtered_signal = self._low_pass_filter(signal, orig_sr, cutoff)
        else:
            filtered_signal = signal

        resampled_signal = librosa.resample(filtered_signal, orig_sr=orig_sr, target_sr=sr)


        # Calculate entropy
        stft = librosa.stft(resampled_signal, n_fft=1024, hop_length=512)
        entropy = self._shannon_entropy(stft)
        times = librosa.frames_to_time(np.arange(len(entropy)), sr=sr, hop_length=512)

        # Normalize entropy
        denom = np.max(entropy) - np.min(entropy)
        entropy = (entropy - np.min(entropy)) / denom if denom != 0 else np.zeros_like(entropy)

        # Smooth entropy
        smoothed_entropy = self._smooth_signal(entropy, window_size=window_size)

        # Detect flat segments
        flat_segments = self.detect_flat_segments(smoothed_entropy, times, min_duration, segments_merge_gap, segments_std_threshold, flucturation_frame_size)

        entropy_data = {
            "times": times[:len(smoothed_entropy)],
            "smoothed_entropy": smoothed_entropy,
            "flat_segments": flat_segments
        }

 
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)


        entropy_data['base64_image'] = plot_entropy(ax=ax,
            total_duration=entropy_data['times'][-1],
            entropy_data= entropy_data,
            flat_segments=flat_segments,
            legend_size=legend_size,
            return_base64=return_base64
        )

        if fig_save_path:
            plt.savefig(fig_save_path, dpi=600, bbox_inches="tight")
        
        if visualize:
            plt.show()
    
        return entropy_data
