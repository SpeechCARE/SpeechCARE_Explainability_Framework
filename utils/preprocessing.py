import numpy as np
import librosa
import scipy.signal
import os
import soundfile as sf  # Add soundfile for saving audio
from typing import Tuple, Optional, Union, List
from tqdm.notebook import tqdm

class Preprocessing:
    """
    A collection of speech preprocessing techniques for AD detection models.
    Includes methods for speech normalization, spectral subtraction, and cepstral mean normalization.
    """

    def __init__(self, sr: int = 16000):
        """
        Initialize the speech preprocessing class.

        Args:
            sr: Sample rate of the audio files (default: 16000 Hz)
        """
        self.sr = sr

    def rms_normalization(self, audio: np.ndarray, target_dBFS: float = -23.0) -> np.ndarray:
        """
        Normalize the audio to a target RMS power in dBFS.

        Args:
            audio: Input audio signal
            target_dBFS: Target RMS power in dBFS (default: -23.0, broadcast standard)

        Returns:
            Normalized audio signal
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))

        # Convert target dBFS to linear gain
        target_rms = 10 ** (target_dBFS / 20.0)

        # Calculate gain factor
        if rms > 0:
            gain = target_rms / rms
        else:
            gain = 1.0  # Avoid division by zero

        # Apply gain
        normalized_audio = audio * gain

        # Clip to avoid distortion
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

        return normalized_audio

    def peak_normalization(self, audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """
        Normalize the audio to a target peak amplitude.

        Args:
            audio: Input audio signal
            target_peak: Target peak amplitude (default: 0.95 to avoid clipping)

        Returns:
            Normalized audio signal
        """
        # Find the absolute peak
        peak = np.max(np.abs(audio))

        # Calculate gain factor
        if peak > 0:
            gain = target_peak / peak
        else:
            gain = 1.0  # Avoid division by zero

        # Apply gain
        normalized_audio = audio * gain

        return normalized_audio

    def spectral_subtraction(self,
                            audio: np.ndarray,
                            noise_estimate: Optional[np.ndarray] = None,
                            noise_seconds: float = 0.5,
                            frame_length: int = 512,
                            hop_length: int = 128,
                            alpha: float = 2.0,
                            beta: float = 0.01) -> np.ndarray:
        """
        Apply spectral subtraction for noise reduction.

        Args:
            audio: Input audio signal
            noise_estimate: Pre-computed noise profile (if None, uses first noise_seconds of audio)
            noise_seconds: Seconds of audio to use for noise estimation (if noise_estimate is None)
            frame_length: STFT frame length
            hop_length: STFT hop length
            alpha: Over-subtraction factor to reduce musical noise
            beta: Spectral floor parameter to avoid negative spectrum

        Returns:
            Noise-reduced audio signal
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
        stft_mag = np.abs(stft)
        stft_phase = np.angle(stft)

        # Get noise profile
        if noise_estimate is None:
            noise_length = int(self.sr * noise_seconds)
            if len(audio) > noise_length:
                noise_stft = librosa.stft(audio[:noise_length], n_fft=frame_length, hop_length=hop_length)
                noise_profile = np.mean(np.abs(noise_stft)**2, axis=1)
            else:
                # If audio is too short, use a fraction of it for noise estimation
                frac = min(0.2, 0.5 * len(audio) / self.sr)
                noise_length = int(self.sr * frac)
                noise_stft = librosa.stft(audio[:noise_length], n_fft=frame_length, hop_length=hop_length)
                noise_profile = np.mean(np.abs(noise_stft)**2, axis=1)
        else:
            noise_stft = librosa.stft(noise_estimate, n_fft=frame_length, hop_length=hop_length)
            noise_profile = np.mean(np.abs(noise_stft)**2, axis=1)

        # Reshape noise profile for broadcasting
        noise_profile = noise_profile.reshape(-1, 1)

        # Apply spectral subtraction
        subtracted_power = stft_mag**2 - alpha * noise_profile

        # Apply spectral floor
        floor = beta * stft_mag**2
        subtracted_power = np.maximum(subtracted_power, floor)

        # Convert back to magnitude
        subtracted_mag = np.sqrt(subtracted_power)

        # Reconstruct complex STFT
        subtracted_stft = subtracted_mag * np.exp(1j * stft_phase)

        # Inverse STFT
        denoised_audio = librosa.istft(subtracted_stft, hop_length=hop_length, length=len(audio))

        return denoised_audio

    def cepstral_mean_normalization(self,
                                   audio: np.ndarray,
                                   n_mfcc: int = 13,
                                   frame_length: int = 512,
                                   hop_length: int = 128,
                                   cmvn_type: str = 'global') -> np.ndarray:
        """
        Apply Cepstral Mean (and Variance) Normalization.

        Args:
            audio: Input audio signal
            n_mfcc: Number of MFCCs to compute
            frame_length: Frame length
            hop_length: Hop length between frames
            cmvn_type: Type of normalization ('global' or 'sliding_window')

        Returns:
            Normalized audio signal
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc,
                                    n_fft=frame_length, hop_length=hop_length)

        if cmvn_type == 'global':
            # Global CMN: Subtract mean from each coefficient
            mfccs_norm = mfccs - np.mean(mfccs, axis=1, keepdims=True)

            # Optional: Also normalize variance (CMVN)
            mfccs_norm = mfccs_norm / (np.std(mfccs_norm, axis=1, keepdims=True) + 1e-10)

        elif cmvn_type == 'sliding_window':
            # Sliding window CMN (more adaptive to changing conditions)
            window_size = min(200, mfccs.shape[1])  # 2 seconds window or less
            mfccs_norm = np.zeros_like(mfccs)

            for i in range(mfccs.shape[1]):
                start = max(0, i - window_size//2)
                end = min(mfccs.shape[1], i + window_size//2)
                window = mfccs[:, start:end]
                mfccs_norm[:, i] = mfccs[:, i] - np.mean(window, axis=1)
                # Also normalize variance
                mfccs_norm[:, i] = mfccs_norm[:, i] / (np.std(window, axis=1) + 1e-10)
        else:
            raise ValueError(f"Unknown CMVN type: {cmvn_type}")

        # Rather than converting back to audio (which isn't typically done with CMVN),
        # we would feed these normalized MFCCs directly to our model.
        # For demonstration purposes, we'll convert to audio using inverse DCT.

        # This is just a demonstration - not typically done
        mel_spectrum = librosa.feature.inverse.mfcc_to_mel(mfccs_norm, n_mels=128)
        mel_spectrum = np.exp(mel_spectrum)  # Convert from log scale
        audio_reconst = librosa.feature.inverse.mel_to_audio(mel_spectrum,
                                                           sr=self.sr,
                                                           n_fft=frame_length,
                                                           hop_length=hop_length,
                                                           length=len(audio))

        return audio_reconst

    def process_directory(self,
                        input_dir: str,
                        output_dir: str,
                        methods: List[str],
                        output_extension: str = '.wav') -> None:
        """
        Process all audio files in a directory using a specified list of methods in order.

        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save processed files
            methods: List of methods to apply in order (e.g., ['rms_normalization', 'spectral_subtraction', 'cepstral_mean_normalization'])
            output_extension: File extension for the output audio files (default: '.wav')
        """
        os.makedirs(output_dir, exist_ok=True)

        audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]

        with tqdm(total=len(audio_files), desc="Processing files", unit="file") as pbar:
            for filename in audio_files:
                pbar.set_description(f"Processing {filename} (Methods: {methods})")

                file_path = os.path.join(input_dir, filename)

                # Load audio
                audio, sr = librosa.load(file_path, sr=self.sr)

                # Apply processing steps in specified order
                processed_audio = audio.copy()

                for method in methods:
                    if method == 'rms_normalization':
                        processed_audio = self.rms_normalization(processed_audio)
                    elif method == 'peak_normalization':
                        processed_audio = self.peak_normalization(processed_audio)
                    elif method == 'spectral_subtraction':
                        processed_audio = self.spectral_subtraction(processed_audio)
                    elif method == 'cepstral_mean_normalization':
                        processed_audio = self.cepstral_mean_normalization(processed_audio)
                    else:
                        raise ValueError(f"Unknown processing method: {method}")

                # Save processed audio
                base_filename, _ = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{base_filename}{output_extension}")
                sf.write(output_path, processed_audio, self.sr)
                pbar.update(1)