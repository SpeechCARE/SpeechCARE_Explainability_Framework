from acoustic_module.PauseExtraction import PauseExtraction

def detect_pauses_from_audio(
    config,
    audio_path,
    sr=16000,
    energy_threshold=0.001,
    min_pause_duration=0.1,
    expansion_threshold=0.03,
    marked=True,
    refined=True,
    verbose=True
):
    """
    Detects pauses in an audio file using configurable thresholds.

    Args:
        config (dict): Configuration settings for pause extraction.
        audio_path (str): Path to the preprocessed audio file.
        sr (int): Sampling rate (default: 16000).
        energy_threshold (float): RMS threshold for detecting silence.
        min_pause_duration (float): Minimum duration (in seconds) to consider a pause.
        expansion_threshold (float): Temporal expansion around detected pauses.
        marked (bool): Whether to mark significant/informative pauses.
        refined (bool): Whether to refine pause boundaries.
        verbose (bool): Whether to print pause count summary.

    Returns:
        List[Tuple]: A list of detected pauses with metadata.
    """
    pause_extractor = PauseExtraction(config, audio_path)
    pauses = pause_extractor.extract_pauses(
        sr=sr,
        energy_threshold=energy_threshold,
        min_pause_duration=min_pause_duration,
        expansion_threshold=expansion_threshold,
        marked=marked,
        refined=refined
    )

    if verbose:
        print(f"✅ Detected {len(pauses)} pauses in: {audio_path}")

    return pauses
