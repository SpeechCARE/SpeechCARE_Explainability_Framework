import json
import string
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import whisperx
from typing import List, Tuple, Optional,Dict
import stanza
import spacy

nlp = stanza.Pipeline(lang="en", processors="tokenize,pos")
nlp_spacy = spacy.load("en_core_web_sm")

class PauseExtraction:
    def __init__(self, config, audio_file: str, word_segments: Optional[str] = None):
        """
        Initialize the PauseExtraction class.

        Args:
            audio_file (str): Path to the audio file.
            word_segments (Optional[str]): Path to the word segments JSON file (if available).
        """
        self.audio_file = audio_file
        self.word_segments = word_segments
        self.device = config.device 
        self.batch_size = config.batch_size
        self.compute_type = config.compute_type
        self.model_id = config.model_id

    def extract_word_segments(self) -> dict:
        """
        Extract word segments from the audio file using WhisperX.

        Returns:
            dict: Transcribed and aligned word segments.
        """
        if not self.word_segments:
            model = whisperx.load_model(self.model_id, self.device, compute_type=self.compute_type)
            audio = whisperx.load_audio(self.audio_file)
            result = model.transcribe(self.audio_file, batch_size=self.batch_size)
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=True)
        else:
            with open(self.word_segments, "r") as f:
                result = json.load(f)
        return result
    
    def extract_pauses(
        self,
        energy_threshold: float = 0.001,
        min_pause_duration: float = 0.2,
        expansion_threshold: float = 0.03,
        sr: int = 16000,
        pause_threshold: float = 0.3,
        context_window_size: int = 6,
        marked: bool = True,
        refined: bool = True
    ) -> List[Tuple[float, float, str, str, str, str, str, str, bool]]:
        """
        Extract and process pauses from word segments with contextual information.

        Args:
            energy_threshold: RMS energy threshold for silence detection (default: 0.001)
            min_pause_duration: Minimum duration to consider as pause in seconds (default: 0.2)
            expansion_threshold: Threshold for pause boundary expansion in seconds (default: 0.03)
            sr: Sampling rate in Hz (default: 16000)
            pause_threshold: Minimum gap between words to consider as pause in seconds (default: 0.3)
            context_window_size: Number of words to include before/after pause for context (default: 6)
            marked: Whether to mark important pauses (default: True)
            refined: Whether to refine pause boundaries using energy (default: True)

        Returns:
            List of tuples containing:
            - start_time: Pause start time in seconds
            - end_time: Pause end time in seconds
            - pre_word: Word before pause
            - pre_pos: POS tag before pause
            - post_word: Word after pause
            - post_pos: POS tag after pause
            - pre_context: Context before pause
            - post_context: Context after pause
            - is_important: Whether pause is marked important
        """
        # 1. Get word segments and POS tags
        word_segments = self.extract_word_segments()
        tagged_words = self.get_pos_tagged_words(word_segments)
        
        # 2. Extract raw pauses
        raw_pauses = self._find_raw_pauses(
            tagged_words,
            pause_threshold=pause_threshold,
            window_size=context_window_size
        )
        
        # 3. Process pauses
        if marked:
            raw_pauses = self.mark_pauses(raw_pauses)
        
        if refined:
            raw_pauses = self.refine_pauses(
                raw_pauses,
                sr=sr,
                energy_threshold=energy_threshold,
                min_pause_duration=min_pause_duration,
                expansion_threshold=expansion_threshold
            )
        
        return raw_pauses

    def _find_raw_pauses(
        self,
        tagged_words: List[Dict],
        pause_threshold: float,
        window_size: int
    ) -> List[Tuple]:
        """
        Identify pauses between words based on temporal gaps.
        
        Args:
            tagged_words: List of words with POS tags and timings
            pause_threshold: Minimum gap duration to consider as pause
            window_size: Context window size in words
            
        Returns:
            List of raw pause tuples
        """
        pauses = []
        
        for i in range(len(tagged_words) - 1):
            gap_duration = tagged_words[i + 1]['start'] - tagged_words[i]['end']
            
            if gap_duration > pause_threshold:
                # Get context window
                start_idx = max(0, i - window_size)
                end_idx = min(len(tagged_words), i + window_size + 1)
                
                # Build context phrases
                pre_context = " ".join(
                    w['word'].lower() 
                    for w in tagged_words[start_idx:i]
                )
                post_context = " ".join(
                    w['word'].lower() 
                    for w in tagged_words[i + 1:end_idx]
                )
                
                # Create pause tuple
                pause = (
                    tagged_words[i]['end'], 
                    tagged_words[i + 1]['start'],
                    tagged_words[i]['word'],
                    tagged_words[i]['POS'],
                    tagged_words[i + 1]['word'],
                    tagged_words[i + 1]['POS'],
                    pre_context,
                    post_context
                )
                pauses.append(pause)
        
        return pauses
    
    def get_pps(self, doc):
        "Function to get PPs from a parsed document."
        pps = []
        for token in doc:
            if token.pos_ == 'ADP':
                pp = ' '.join([tok.orth_ for tok in token.subtree])
                pps.append(pp)
        return pps

    def mark_pauses(self, pauses: List[Tuple]) -> List[Tuple]:
        """
        Mark pauses based on linguistic rules.

        Args:
            pauses (List[Tuple]): List of pauses with start, end, and context information.

        Returns:
            List[Tuple]: List of pauses with additional marking information.
        """
        conjunctions = {"and", "but", "or", "yet", "so", "for", "nor", "although", "because", "since", "unless", "while", "whereas", "though", "if", "when", "before", "after"}
        uncertainty_phrases = {"may", "might", "could", "can", "guess", "unsure", "probably", "perhaps", "possibly", "im not sure", "i think", "it seems", "that i see"}
        noun_tags = {"NOUN", "NN", "NNS", "NNP", "NNPS"}

        marked_pauses = []

        for pause in pauses:
            start_pause, end_pause, prev_word, prev_word_POS, next_word, next_word_POS, prev_phrase, next_phrase = pause
            mark = None

            if prev_word.lower() in conjunctions or next_word.lower() in conjunctions:
                mark = "Pause around conjunction"
            elif any(phrase in prev_phrase.lower() or phrase in next_phrase.lower() for phrase in uncertainty_phrases):
                mark = "Pause around uncertainty word"
            elif next_word_POS in noun_tags:
                mark = "Pause before noun"
            else:
                doc = nlp_spacy(next_phrase.lower())

                prep_phrases = self.get_pps(doc)
                prep_phrases = [phrase for phrase in prep_phrases if phrase.startswith(next_word.lower())]
                noun_phrases = [chunk.text for chunk in doc.noun_chunks]
                noun_phrases = [phrase for phrase in noun_phrases if phrase.startswith(next_word.lower())]
                
                if noun_phrases:
                    mark = "Pause before noun phrase"
                elif prep_phrases:
                    mark = "Pause before prepositional phrase"


            marked_pauses.append((start_pause, end_pause, prev_word, prev_word_POS, next_word, next_word_POS, mark))

        return marked_pauses

    def plot_pauses(self, log_mel_spectrogram: np.ndarray, sr: int, y: np.ndarray, pauses: List[Tuple], plot: bool = True, save_path: Optional[str] = None):
        """
        Plot pauses on a given mel spectrogram and either display or save the figure.

        Args:
            log_mel_spectrogram (np.ndarray): Log mel spectrogram of the audio.
            sr (int): Sample rate of the audio.
            y (np.ndarray): Audio waveform.
            pauses (List[Tuple]): List of pauses to plot.
            plot (bool): Whether to display the plot (default: True).
            save_path (Optional[str]): Path to save the plot (default: None).
        """
        plt.figure(figsize=(24, 4))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel", cmap="viridis")
        max_mel = 8500

        for start, end, _, _, _, _, mark in pauses:
            color = "red" if mark else "yellow"
            plt.plot([start, start, end, end, start], [0, max_mel, max_mel, 0, 0], color=color, linewidth=2)

        duration = librosa.get_duration(y=y, sr=sr)
        time_ticks = np.arange(0, duration, step=0.3)
        plt.xticks(time_ticks, labels=[f"{t * 1000:.0f}" for t in time_ticks], fontsize=8, rotation=45)
        plt.ylim(0, max_mel)
        plt.xlabel("Time (ms)")
        plt.ylabel("Mel Frequency (Hz)")
        plt.title("Log Mel Spectrogram with Detected Pauses (Outlined)")

        # Save the plot if save_path is provided
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Plot saved to {save_path}")

        # Display the plot if plot is True
        if plot:
            plt.show()

        plt.close()

    def refine_pauses(self, pauses: List[Tuple], sr: int, energy_threshold: float, min_pause_duration: float, expansion_threshold: float) -> List[Tuple]:
        """
        Refine pauses based on energy thresholds and duration.

        Args:
            pauses (List[Tuple]): List of pauses to refine.
            sr (int): Sample rate of the audio.
            energy_threshold (float): Energy threshold for pause refinement.
            min_pause_duration (float): Minimum duration for a valid pause.
            expansion_threshold (float): Threshold for expanding pauses.

        Returns:
            List[Tuple]: Refined pauses.
        """
        refined_pauses = []

        for pause in pauses:
            start_time, end_time, prev_word, prev_word_pos, next_word, next_word_pos, mark = pause
            audio_segment, _ = librosa.load(self.audio_file, sr=sr, offset=start_time, duration=end_time - start_time)
            energy = self.compute_energy(audio_segment)
            times = librosa.times_like(energy, sr=sr, hop_length=512) + start_time

            high_energy_indices = np.where(energy > energy_threshold)[0]
            if len(high_energy_indices) == 0:
                refined_pauses.append(pause)
                continue

            continuous_high = self.extract_continuous_sequences(high_energy_indices)
            continuous_high = [item for item in continuous_high if len(item) > 1]
            high_energy_times = [times[c_high] for c_high in continuous_high]
            exclude_intervals = [(high_time[0], high_time[-1]) for high_time in high_energy_times]

            valid_pauses = self.get_valid_intervals(start_time, end_time, exclude_intervals, min_pause_duration, prev_word, prev_word_pos, next_word, next_word_pos, mark, expansion_threshold, sr, energy_threshold)
            refined_pauses.extend(valid_pauses)

        return refined_pauses

    @staticmethod
    def compute_energy(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Compute short-term energy of an audio signal.

        Args:
            audio (np.ndarray): Audio signal.
            frame_length (int): Frame length for energy computation.
            hop_length (int): Hop length for energy computation.

        Returns:
            np.ndarray: Energy values.
        """
        return librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    @staticmethod
    def extract_continuous_sequences(arr: List[int]) -> List[List[int]]:
        """
        Extract continuous sequences from a list of indices.

        Args:
            arr (List[int]): List of indices.

        Returns:
            List[List[int]]: List of continuous sequences.
        """
        sequences = []
        temp = [arr[0]]

        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1] + 1:
                temp.append(arr[i])
            else:
                sequences.append(temp)
                temp = [arr[i]]
        sequences.append(temp)
        return sequences

    def get_valid_intervals(self, start_base: float, end_base: float, exclude_intervals: List[Tuple], min_pause_duration: float, prev_word: str, prev_word_pos: str, next_word: str, next_word_pos: str, mark: str, expansion_threshold: float, sr: int, energy_threshold: float) -> List[Tuple]:
        """
        Get valid pause intervals after excluding high-energy regions.

        Args:
            start_base (float): Start time of the pause.
            end_base (float): End time of the pause.
            exclude_intervals (List[Tuple]): Intervals to exclude.
            min_pause_duration (float): Minimum duration for a valid pause.
            prev_word (str): Previous word.
            prev_word_pos (str): POS tag of the previous word.
            next_word (str): Next word.
            next_word_pos (str): POS tag of the next word.
            mark (str): Mark for the pause.
            expansion_threshold (float): Threshold for expanding pauses.
            sr (int): Sample rate of the audio.
            energy_threshold (float): Energy threshold for pause refinement.

        Returns:
            List[Tuple]: Valid pause intervals.
        """
        valid_intervals = []
        current_start = start_base

        for exclude_start, exclude_end in sorted(exclude_intervals):
            if current_start < exclude_start:
                valid_intervals.append((current_start, exclude_start, prev_word, prev_word_pos, next_word, next_word_pos, mark))
            current_start = max(current_start, exclude_end)

        if current_start < end_base:
            valid_intervals.append((current_start, end_base, prev_word, prev_word_pos, next_word, next_word_pos, mark))

        valid_intervals = [interval for interval in valid_intervals if (interval[1] - interval[0]) > min_pause_duration]
        expanded_intervals = []

        for interval in valid_intervals:
            proposed_start = interval[0] - expansion_threshold
            proposed_end = interval[1] + expansion_threshold

            pre_segment, post_segment = None, None

            try:
                pre_segment, _ = librosa.load(self.audio_file, sr=sr, offset=proposed_start, duration=interval[0] - proposed_start)
            except Exception:
                proposed_start = interval[0]

            try:
                post_segment, _ = librosa.load(self.audio_file, sr=sr, offset=interval[1], duration=proposed_end - interval[1])
            except Exception:
                proposed_end = interval[1]

            if pre_segment is not None:
                pre_energy = self.compute_energy(pre_segment)
                if np.all(pre_energy < energy_threshold):
                    interval = (proposed_start, interval[1], prev_word, prev_word_pos, next_word, next_word_pos, mark)

            if post_segment is not None:
                post_energy = self.compute_energy(post_segment)
                if np.all(post_energy < energy_threshold):
                    interval = (interval[0], proposed_end, prev_word, prev_word_pos, next_word, next_word_pos, mark)

            expanded_intervals.append(interval)

        return expanded_intervals

    @staticmethod
    def get_pos_tagged_words(data: dict) -> List[dict]:
        """
        Perform POS tagging on the word segments.

        Args:
            data (dict): Word segments from WhisperX.

        Returns:
            List[dict]: List of words with POS tags.
        """
        exclude = set(string.punctuation) | {"’"}
        text = " ".join([item['text'] for item in data["segments"]])
        text = ''.join(ch for ch in text if ch not in exclude)
        doc = nlp(text)
        stanza_tokens = [(word.text, word.upos) for sentence in doc.sentences for word in sentence.words]
        stanza_tokens = PauseExtraction.concatenate_part_tuples(stanza_tokens)

        words = []
        for word_info in data["word_segments"]:
            text = ''.join(ch for ch in word_info['word'] if ch not in exclude)
            start = word_info['start']
            end = word_info['end']
            tags = [k for k in stanza_tokens if k[0] == text]
            closest_index = min(range(len(tags)), key=lambda i: abs(i - len(words)))
            words.append({"word": text, "start": start, "end": end, "POS": tags[closest_index][1]})

        return words

    @staticmethod
    def concatenate_part_tuples(tuples_list: List[Tuple]) -> List[Tuple]:
        """
        Concatenate part tuples for POS tagging.

        Args:
            tuples_list (List[Tuple]): List of (word, POS) tuples.

        Returns:
            List[Tuple]: Concatenated tuples.
        """
        result = []
        bad_parts = ["nt"]

        for word, tag in tuples_list:
            if tag == "PART" and word in bad_parts:
                if result:
                    prev_word, prev_tag = result[-1]
                    new_word = prev_word + word
                    result[-1] = (new_word, prev_tag)
            else:
                result.append((word, tag))
        return result