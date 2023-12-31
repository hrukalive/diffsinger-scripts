#%%
import librosa
from pathlib import Path
import parselmouth as pm
import textgrid
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from copy import deepcopy
import click
#%%
f0_min = 40
f0_max = 1100
br_len = 0.15
br_db = -60.0
br_centroid = 2000.0
time_step = 0.005
min_space = 0.04
voicing_thresh_vowel = 0.45
voicing_thresh_breath = 0.6
br_win_sz = 0.05
sil_token = 'SP'
br_token = 'AP'

from .utils import lab_to_tg_tier

def clear_sils(tier, tokens: set):
    result = []
    for ph in tier:
        if ph.mark in tokens:
            if result and result[-1][0] is None:
                assert result[-1][2] == ph.minTime
                result[-1][2] = ph.maxTime
            else:
                result.append([None, ph.minTime, ph.maxTime])
        else:
            result.append([ph.mark, ph.minTime, ph.maxTime])
    tier = textgrid.IntervalTier(name=tier.name)
    for ph in result:
        tier.add(ph[1], ph[2], ph[0])
    return tier

def tg_tier_to_lab(tier):
    result = []
    last = None
    for ph in tier:
        assert ph.mark
        s, e = round(ph.minTime * 10000000), round(ph.maxTime * 10000000)
        if last and s != last:
            raise Exception("Time mismatch")
        last = e
        result.append((s, e, ph.mark))
    return result

#%%
label_path = Path(r'L:\wav_merge')
wav_path = Path(r'L:\wav_merge')
output_label_path = Path(r'L:\wav_merge\tg_fix')
#%%
def fix_br_in_tier(tier, wav, orig_sr):
    tier = deepcopy(tier)
    sr = 24000
    peaks = []
    for i in range(0, len(wav), 2048):
        peaks.append(np.max(np.abs(wav[i:i + 2048])))
    peak = np.percentile(peaks, 99)
    wav /= peak

    sound = pm.Sound(wav)
    f0_voicing_breath = sound.to_pitch_ac(
        time_step=time_step,
        voicing_threshold=voicing_thresh_breath,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    ).selected_array['frequency']
    hop_size = int(time_step * sr)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=librosa.resample(wav, orig_sr=orig_sr, target_sr=sr), sr=sr,
        n_fft=2048, hop_length=hop_size
    ).squeeze(0)

    # Detect aspiration
    i = 0
    while i < len(tier):
        phone = tier[i]
        if phone.mark is not None and phone.mark != '':
            i += 1
            continue
        if phone.maxTime - phone.minTime < br_len:
            i += 1
            continue
        ap_ranges = []
        br_start = None
        win_pos = phone.minTime
        while win_pos + br_win_sz <= phone.maxTime:
            all_noisy = (f0_voicing_breath[
                            int(win_pos / time_step): int((win_pos + br_win_sz) / time_step)] < f0_min).all()
            rms_db = 20 * np.log10(
                np.clip(sound.get_rms(from_time=win_pos, to_time=win_pos + br_win_sz), a_min=1e-12, a_max=1))
            # print(win_pos, win_pos + br_win_sz, all_noisy, rms_db)
            if all_noisy and rms_db >= br_db:
                if br_start is None:
                    br_start = win_pos
            else:
                if br_start is not None:
                    br_end = win_pos + br_win_sz - time_step
                    if br_end - br_start >= br_len:
                        centroid = spectral_centroid[int(br_start / time_step): int(br_end / time_step)].mean()
                        if centroid >= br_centroid:
                            ap_ranges.append((br_start, br_end))
                    br_start = None
                    win_pos = br_end
            win_pos += time_step
        if br_start is not None:
            br_end = win_pos + br_win_sz - time_step
            if br_end - br_start >= br_len:
                centroid = spectral_centroid[int(br_start / time_step): int(br_end / time_step)].mean()
                if centroid >= br_centroid:
                    ap_ranges.append((br_start, br_end))
        # print(ap_ranges)
        if len(ap_ranges) == 0:
            i += 1
            continue
        tier.removeInterval(phone)
        if phone.minTime < ap_ranges[0][0]:
            tier.add(minTime=phone.minTime, maxTime=ap_ranges[0][0], mark=None)
            i += 1
        for k, ap in enumerate(ap_ranges):
            if k > 0:
                tier.add(minTime=ap_ranges[k - 1][1], maxTime=ap[0], mark=None)
                i += 1
            tier.add(minTime=ap[0], maxTime=min(phone.maxTime, ap[1]), mark=br_token)
            i += 1
        if ap_ranges[-1][1] < phone.maxTime:
            tier.add(minTime=ap_ranges[-1][1], maxTime=phone.maxTime, mark=None)
            i += 1

    # Remove short spaces
    i = 0
    while i < len(tier):
        phone = tier[i]
        if phone.mark is not None and phone.mark != '':
            i += 1
            continue
        if phone.maxTime - phone.minTime >= min_space:
            phone.mark = sil_token
            i += 1
            continue
        if i == 0:
            if len(tier) >= 2:
                tier[i + 1].minTime = phone.minTime
                tier.removeInterval(phone)
            else:
                break
        elif i == len(tier) - 1:
            if len(tier) >= 2:
                tier[i - 1].maxTime = phone.maxTime
                tier.removeInterval(phone)
            else:
                break
        else:
            tier[i - 1].maxTime = tier[i + 1].minTime = (phone.minTime + phone.maxTime) / 2
            tier.removeInterval(phone)
    round_tier = textgrid.IntervalTier(name=tier.name)
    for ph in tier:
        round_tier.add(round(ph.minTime, 7), round(ph.maxTime, 7), ph.mark)
    return round_tier
#%%
for label_fn in tqdm(natsorted(label_path.glob('*.lab'))):
    wav_fn = wav_path / label_fn.with_suffix('.wav').name
    out_fn = output_label_path / label_fn.name
    if out_fn.exists():
        continue
    assert wav_fn.exists()
    phones = clear_sils(lab_to_tg_tier(label_fn), {'sil', 'pau'})

    y, sr = librosa.load(wav_fn, sr=44100, mono=True)
    fixed = fix_br_in_tier(phones, y, sr)

    with open(out_fn, 'w') as f:
        f.write('\n'.join(f'{s} {e} {p}' for s, e, p in tg_tier_to_lab(fixed)))
# %%
for label_fn in tqdm(natsorted(label_path.glob('*.TextGrid'))):
    wav_fn = wav_path / label_fn.with_suffix('.wav').name
    out_fn = output_label_path / label_fn.name
    if out_fn.exists():
        continue
    assert wav_fn.exists()
    tg = textgrid.TextGrid().fromFile(str(label_fn))
    phones = clear_sils(tg[0], {'SP', 'AP'})

    y, sr = librosa.load(wav_fn, sr=44100, mono=True)
    fixed = fix_br_in_tier(phones, y, sr)

    tg_out = textgrid.TextGrid()
    tg_out.append(fixed)
    for tier in tg[1:]:
        tg_out.append(tier)
    tg_out.write(out_fn)
# %%
