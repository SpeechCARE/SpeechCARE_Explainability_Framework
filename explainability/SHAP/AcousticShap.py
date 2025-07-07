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
from explainability.plotting.explainability_plotting import plot_SHAP_highlighted_spectrogram,plot_spectrogram

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
        input_values,
        baseline_type='zeros'
      ):

        device = next(self.model.parameters()).device
        input_values = input_values.clone().detach().to(device)
        # print(input_values.shape)

        self.model.eval()
        with torch.no_grad():
            predictions, embeddings = self.model.speech_only_forward(input_values, return_embeddings=True)

        segments_tensor = input_values.squeeze(0)

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
            "logits": predictions,
            "predicted_label":torch.argmax(predictions, dim=1).item()
        }

    def get_speech_spectrogram(
        self,
        audio_path: str,
        demography_info: Any,
        config: dict,
        *,
        spectrogram_type: str = "original",
        formants_to_plot: Optional[List[str]] = None,
        pauses: Optional[List[Tuple[float, float]]] = None,
        sr: int = None,
        segment_length: float = 5,
        overlap: float = 0.2,
        frame_duration: float = 0.3,
        baseline_type: str = 'zeros',
        fig_save_dir: Optional[str] = 'spectrogram_figs',
        plot: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Enhanced spectrogram analysis with spectral entropy and formant tracking support.

        Args:
            audio_path: Path to audio file
            demography_info: Demographic information for model
            config: Model configuration dictionary
            spectrogram_type: Type of spectrogram ("original" or "shap")
            formants_to_plot: List of formant names to plot (e.g., ['F1', 'F2'])
            pauses: List of pause intervals as (start, end) tuples
            sr: Target sampling rate (uses default if None)
            segment_length: SHAP segment length in seconds
            overlap: SHAP segment overlap ratio
            frame_duration: SHAP merge frame duration in seconds
            baseline_type: SHAP baseline type
            fig_save_dir: Directory to save output figure
            plot: Whether to display the plot

        Returns:
            Returns vary based on requested analyses:
            - Spectrogram only (default)
            - (spectrogram, formant_values) if formants_to_plot is not empty

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
        self._generate_spectrogram(
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
            fig_save_path = os.path.join(fig_save_dir,'spectrogram.png'),
            plot=plot
        )

  

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
        fig_save_path:str,
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
                fig_save_path=fig_save_path,
                plot=plot
            )
        else:

            acoustic_shap_values = self.calculate_speech_shap_values(
                audio_path,
                segment_length=segment_length,
                overlap=overlap,
                target_sr=sr,
                baseline_type=baseline_type,
            )["shap_values"]

            return self.visualize_shap_spectrogram(
                ax=ax,
                audio_path=audio_path,
                shap_values=acoustic_shap_values['shap_values'],
                label=acoustic_shap_values['predicted_label'],
                sr=sr,
                segment_length=segment_length,
                overlap=overlap,
                merge_frame_duration=frame_duration,
                formants_to_plot=formants,
                pauses=pauses,
                fig_save_path=fig_save_path,
                plot=plot
            )



    def visualize_original_spectrogram(
        self,
        audio_path,
        sr=16000,
        formants_to_plot=None,
        pauses=None,
        hop_length=512,
        fig_save_path=None,
        ax=None,
        plot=False,
        figsize = (10, 4)
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
        fig, ax = None, None
                
        # Only create a figure if we need to plot or save
        if plot or fig_save_path:
            fig, ax = plt.subplots(figsize=figsize)
            plot_spectrogram(
                ax=ax,
                total_duration=None,
                audio_path=audio_path,
                sr=sr,
                hop_length=hop_length,
                pauses=pauses,
                formants_data=formants_to_plot,
                title="Spectrogram"
            )

            if fig_save_path:
                plt.savefig(fig_save_path, dpi=600, bbox_inches="tight")
            
            if plot:
                plt.show()
            else:
                plt.close(fig)  # Close if not displaying to free memory

        return (fig, ax) if (plot or fig_save_path) else None
    


    def visualize_shap_spectrogram(
        self,
        audio_path,
        shap_values,
        label,
        sr=16000,
        hop_length=512,
        segment_length=5,
        overlap=0.2,
        merge_frame_duration=0.3,
        formants_to_plot=None,
        fig_save_path=None,
        pauses = None,
        ax=None,
        plot=False,
        figsize = (10, 4)
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
            Tuple[Optional[plt.Figure], Optional[plt.Axes]]: Figure and axes objects if plot/save is enabled, else None.
        """
      
        
        fig, ax = None, None
                
        # Only create a figure if we need to plot or save
        if plot or fig_save_path:
            fig, ax = plt.subplots(figsize=figsize)
            plot_SHAP_highlighted_spectrogram(
                ax=ax,
                total_duration=None,
                audio_path=audio_path,
                shap_values=shap_values,
                pauses=pauses,
                formants_data=formants_to_plot,
                label=label,
                sr=sr,
                hop_length=hop_length,
                segment_length=segment_length,
                overlap=overlap,
                merge_frame_duration=merge_frame_duration,
            )

            if fig_save_path:
                plt.savefig(fig_save_path, dpi=600, bbox_inches="tight")
            
            if plot:
                plt.show()
            else:
                plt.close(fig)  # Close if not displaying to free memory

        return (fig, ax) if (plot or fig_save_path) else None
    
      