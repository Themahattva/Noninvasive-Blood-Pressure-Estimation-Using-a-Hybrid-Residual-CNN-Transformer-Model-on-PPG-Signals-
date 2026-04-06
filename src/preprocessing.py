import numpy as np
import pywt
from scipy.signal import butter, filtfilt, find_peaks


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def apply_highpass(signal, cutoff=0.5, fs=125, order=2):
    b, a = butter_highpass(cutoff, fs, order)
    return filtfilt(b, a, signal)


def wavelet_denoise(signal, wavelet='db6', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    lam = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, lam, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]


def minmax_normalize(signal):
    eps = 1e-8
    xmin, xmax = signal.min(), signal.max()
    return (signal - xmin) / (xmax - xmin + eps)


def detect_peaks(signal, fs=125, min_hr=30, max_hr=120):
    min_dist = int(fs * 60 / max_hr)
    peaks, _ = find_peaks(signal, distance=min_dist)
    return peaks


def is_valid_segment(ppg, abp, fs=125, cfg=None):
    if cfg is None:
        cfg = {}
    min_std_ppg = cfg.get('min_std_ppg', 0.05)
    min_std_abp = cfg.get('min_std_abp', 5.0)
    min_peaks = cfg.get('min_peaks', 10)
    abp_min = cfg.get('abp_min', 20)
    abp_max = cfg.get('abp_max', 200)
    ppg_amp_max = cfg.get('ppg_amp_max', 4.0)
    ppg_mean_min = cfg.get('ppg_mean_min', 0.1)

    if np.std(ppg) < min_std_ppg:
        return False
    if np.std(abp) < min_std_abp:
        return False
    peaks, _ = find_peaks(ppg, distance=int(0.5 * fs))
    if len(peaks) < min_peaks:
        return False
    if abp.min() < abp_min or abp.max() > abp_max:
        return False
    if ppg.max() >= ppg_amp_max or ppg.mean() <= ppg_mean_min:
        return False
    return True


def extract_bp_labels(abp_window, fs=125):
    min_dist = 50
    sys_peaks, _ = find_peaks(abp_window, distance=min_dist)
    dia_troughs, _ = find_peaks(-abp_window, distance=min_dist)
    if len(sys_peaks) == 0 or len(dia_troughs) == 0:
        return None, None
    sbp = float(np.mean(abp_window[sys_peaks]))
    dbp = float(np.mean(abp_window[dia_troughs]))
    return sbp, dbp


def preprocess_ppg(ppg_raw, fs=125):
    ppg_hp = apply_highpass(ppg_raw, cutoff=0.5, fs=fs)
    ppg_dn = wavelet_denoise(ppg_hp)
    ppg_norm = minmax_normalize(ppg_dn)
    return ppg_norm


def extract_windows(ppg, abp, fs=125, window_size=1250, overlap=625, cfg=None):
    windows_ppg, labels = [], []
    step = window_size - overlap
    n = len(ppg)
    i = 0
    while i + window_size <= n:
        seg_ppg = ppg[i:i + window_size]
        seg_abp = abp[i:i + window_size]
        if is_valid_segment(seg_ppg, seg_abp, fs, cfg):
            sbp, dbp = extract_bp_labels(seg_abp, fs)
            if sbp is not None:
                proc_ppg = preprocess_ppg(seg_ppg, fs)
                windows_ppg.append(proc_ppg)
                labels.append((sbp, dbp))
            i += window_size  # advance past clean segment
        else:
            i += fs * 5  # advance 5 seconds
    return windows_ppg, labels
