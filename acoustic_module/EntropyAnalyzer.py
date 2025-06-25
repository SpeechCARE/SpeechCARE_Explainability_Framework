import numpy as np
from scipy.signal import butter, filtfilt
from typing import List, Tuple, Optional, Dict, Any
import librosa
import matplotlib.pyplot as plt

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
    
    def _low_pass_filter(
        self,
        signal: np.ndarray,
        sr: int,
        cutoff: int,
        order: int = 6
    ) -> np.ndarray:
        """
        Apply low-pass Butterworth filter to the signal.
        
        Args:
            signal: Input audio signal
            sr: Sampling rate
            cutoff: Cutoff frequency
            order: Filter order (default: 6)
            
        Returns:
            Filtered signal
        """
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    
    def _shannon_entropy(self, stft: np.ndarray) -> np.ndarray:
        """
        Calculate Shannon entropy from STFT magnitudes.
        
        Args:
            stft: Short-time Fourier transform of the signal
            
        Returns:
            Array of entropy values
        """
        ps = np.abs(stft)**2
        ps_norm = ps / np.sum(ps, axis=0, keepdims=True)
        ps_norm[ps_norm == 0] = 1e-12  # Avoid log(0)
        return -np.sum(ps_norm * np.log2(ps_norm), axis=0)
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """
        Smooth signal using moving average.
        
        Args:
            signal: Input signal
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed signal
        """
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
    
    def analyze(
        self,
        audio_path: str,
        sr: Optional[int] = None,
        cutoff: Optional[int] = None,
        window_size: int = 15,
        min_duration: float = 5,
        segments_std_threshold: float = 0.4,
        segments_merge_gap: float = 0.5,
        figsize: Tuple[int, int] = (20, 2),
        dpi: int = 200,
        visualize: bool = False,
        filter_signal:bool = False,
    ) -> Dict[str, Any]:
        """
        Complete entropy analysis with optional visualization.
        
        Args:
            audio_path: Path to audio file
            sr: Target sampling rate
            cutoff: Low-pass cutoff frequency
            window_size: Smoothing window size
            min_duration: Minimum flat segment duration
            segments_std_threshold: Maximum standard deviation threshold for 
                                    considering a segment as "flat"
            segments_merge_gap: Maximum time gap between adjacent flat 
                                segments to allow merging
            figsize: Figure dimensions
            dpi: Figure resolution
            visualize: Whether to show visualization
            
        Returns:
            Dictionary containing:
                - times: Time points
                - smoothed_entropy: Smoothed entropy values
                - flat_segments: Detected flat segments as (start, end) tuples
        """
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
        smoothed_entropy = self._smooth_signal(entropy, window_size)
        
        # Detect flat segments
        flat_segments = self.detect_flat_segments(
            smoothed_entropy,
            times[:len(smoothed_entropy)],
            std_threshold=segments_std_threshold,
            merge_gap=segments_merge_gap,
            final_min_duration=min_duration
        )
        
        # Visualization
        if visualize:
            self._visualize_analysis(
                times[:len(smoothed_entropy)],
                smoothed_entropy,
                flat_segments,
                figsize,
                dpi
            )
        
        return {
            "times": times[:len(smoothed_entropy)],
            "smoothed_entropy": smoothed_entropy,
            "flat_segments": flat_segments
        }
    
    def _visualize_analysis(
        self,
        times: np.ndarray,
        smoothed_entropy: np.ndarray,
        flat_segments: List[Tuple[float, float]],
        figsize: Tuple[int, int],
        dpi: int
    ) -> None:
        """
        Visualize the entropy analysis results.
        
        Args:
            times: Time points
            smoothed_entropy: Smoothed entropy values
            flat_segments: Detected flat segments
            figsize: Figure dimensions
            dpi: Figure resolution
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(times, smoothed_entropy, label='Smoothed Entropy')
        
        for start, end in flat_segments:
            plt.axvspan(start, end, color='green', alpha=0.3, label='Flat Segment' if start == flat_segments[0][0] else "")
        
        plt.xlabel('Time (s)')
        plt.ylabel('Entropy')
        plt.title('Entropy Analysis with Flat Segments')
        plt.legend()
        plt.tight_layout()
        plt.show()