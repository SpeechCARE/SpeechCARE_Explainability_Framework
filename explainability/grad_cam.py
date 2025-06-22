import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.ndimage import zoom


def register_gradcam_hook(model, layer, act_dict, grad_dict, name):
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        act_dict[name] = out
        out.register_hook(lambda grad: grad_dict.setdefault(name, grad))
    layer.register_forward_hook(hook_fn)


def compute_gradcam(model, input_tensor, act_dict, grad_dict, layer_name, target_class_idx=None):
    input_tensor = input_tensor.cuda()
    output = model.speech_only_forward(input_tensor)

    if target_class_idx is None:
        target_class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class_idx].backward()

    act = act_dict[layer_name]         # shape: [1, T, C]
    grad = grad_dict[layer_name]       # shape: [1, T, C]

    weights = grad.mean(dim=1, keepdim=True)  # [1, 1, C]
    cam = (weights @ act.transpose(1, 2)).squeeze()  # [T]
    cam = torch.relu(cam).detach().cpu().numpy()
    return cam


def upsample_cam(cam, target_len):
    return zoom(cam, target_len / len(cam))


def normalize_cam(cam, threshold=0.6):
    cam_norm = cam - cam.min()
    cam_norm /= (cam_norm.max() + 1e-6)
    cam_masked = np.ma.masked_where(cam_norm < threshold, cam_norm)
    return cam_masked


def get_spectrogram_axes(spec, hop_length=160, sr=16000):
    n_mels, n_frames = spec.shape
    duration = n_frames * hop_length / sr
    time_axis = np.linspace(0, duration, n_frames)
    freq_axis = np.linspace(0, sr // 2, n_mels)
    return time_axis, freq_axis


def plot_cam_over_spectrogram(spec, cam_masked, time_axis, freq_axis, title="Grad-CAM over Mel-Spectrogram"):
    plt.figure(figsize=(25, 10))

    plt.imshow(spec,
               aspect='auto', origin='lower',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
               cmap='gray_r')

    plt.imshow(cam_masked[np.newaxis, :],
               aspect='auto', origin='lower',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
               cmap='jet', alpha=0.5, interpolation='bilinear')

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    cbar = plt.colorbar(label="Attention Intensity")
    cbar.set_alpha(1)
    plt.tight_layout()
    plt.show()
