
import librosa
import parselmouth
import numpy as np

def compute_log_spectrogram(audio_path,sr=16000, hop_length=512):

    import librosa
    import librosa.display
    import numpy as np

    audio, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=hop_length, power=2.0)

    log_S = librosa.power_to_db(S, ref=np.max)

    return len(audio) / sr,log_S



def compute_formants(audio_path, formants_data):
    
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    f0_values = pitch.selected_array["frequency"]
    f0_values[f0_values == 0] = np.nan
    times = np.arange(0, sound.get_total_duration(), 0.01)
    formant = sound.to_formant_burg(time_step=0.1)

    formant_values = {"F0": f0_values, "F1": [], "F2": [], "F3": []}
    for t in times:
        formant_values["F1"].append(formant.get_value_at_time(1, t))
        formant_values["F2"].append(formant.get_value_at_time(2, t))
        formant_values["F3"].append(formant.get_value_at_time(3, t))
    return pitch.ts(), times, formant_values
