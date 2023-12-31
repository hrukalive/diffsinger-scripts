#%%
import argparse
import math
from collections import deque
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import textgrid
from natsort import natsorted
from tqdm import tqdm

from .utils import lab_to_tg


#%%
def scan_label(tier, sil_symbol, br_symbol):
    errored = False
    prev_time = None
    for l in tier:
        s, e, p = l.minTime, l.maxTime, l.mark
        if s > e:
            print(l)
            errored = True
        if prev_time is not None and s != prev_time:
            if abs(s - prev_time) > 0.005 and s < prev_time:
                print(l)
                errored = True
        if e - s < 0.001:
            print(l)
            errored = True
        if (sil_symbol != 'SP' and p == 'SP') or (br_symbol != 'AP' and p == 'AP'):
            raise Exception("SP/AP are reserved.")
        prev_time = e
    if errored:
        raise Exception("Timing problem detected in label files")
#%%
def read_label(label_fn, wav_fn, sil_symbol, br_symbol):
    wav_sec = round(librosa.get_duration(filename=wav_fn), 6)

    if label_fn.suffix == '.TextGrid':
        tg = textgrid.TextGrid.fromFile(str(label_fn))
    elif label_fn.suffix == '.lab' or label_fn.suffix == '.txt':
        tg = lab_to_tg(label_fn)
    else:
        raise Exception("Unknown file type")
    scan_label(tg[0])
    # Read phonemes
    phones = []
    prev_time = None
    for l in tg[0]:
        s, e, p = l.minTime, l.maxTime, l.mark
        if p == sil_symbol:
            p = 'SP'
        elif p == br_symbol:
            p = 'AP'
        e = min(e, wav_sec)
        if s > wav_sec:
            raise Exception(f'{s} > {wav_sec}')

        if prev_time is not None and s != prev_time:
            if abs(s - prev_time) > 0.05:
                if s < prev_time:
                    raise ValueError(l)
                else:
                    phones.append(['SP', (prev_time, s)])
            else:
                s = prev_time
        if s > e:
            print(label_fn, l, s, e)
            raise Exception()
        prev_time = e
        if p == 'SP' and phones and phones[-1][0] == 'SP' and phones[-1][1][1] == s:
            phones[-1][1] = (phones[-1][1][0], e)
            if phones[-1][1][0] > e:
                print(label_fn, l)
        else:
            phones.append([p, (s, e)])
    # Check against wav length
    if math.isclose(phones[-1][1][1], wav_sec, abs_tol=0.01):
        phones[-1][1] = (phones[-1][1][0], wav_sec)
    elif phones[-1][1][1] < wav_sec:
        phones.append(['SP', (phones[-1][1][1], wav_sec)])
    # Check time consistency
    for ph1, ph2 in zip(phones[:-1], phones[1:]):
        if ph1[1][1] != ph2[1][0]:
            raise Exception("Time mismatch")
    # Merge consecutive sils
    tmp_phones = []
    for ph in phones:
        if tmp_phones and ph[0] == 'SP' and tmp_phones[-1][0] == 'SP':
            tmp_phones[-1][1] = (tmp_phones[-1][1][0], ph[1][1])
        else:
            tmp_phones.append(deepcopy(ph))
    assert len([x for x in phones if x[0] != 'SP']) == len([x for x in tmp_phones if x[0] != 'SP'])
    for ph1, ph2 in zip(tmp_phones[:-1], tmp_phones[1:]):
        if ph1[0] == 'SP' and ph2[0] == 'SP':
            raise Exception("Consecutive sils")
    phones = tmp_phones

    return tg, phones
# %%
def split_to_large_slices(phones):
    large_slices = []
    tmp = []
    for ph in phones:
        tmp.append(ph)
        if ph[0] == 'SP':
            if not (len(tmp) == 1 and tmp[0][0] == 'SP'):
                large_slices.append(deepcopy(tmp))
            tmp = [ph]
    if tmp:
        ph_set = set([x[0] for x in tmp])
        if not (len(ph_set) == 1 and 'SP' in ph_set):
            large_slices.append(deepcopy(tmp))
    for sli in large_slices:
        ph_set = set([x[0] for x in sli])
        if len(ph_set) == 1 and 'SP' in ph_set:
            raise Exception("SP only slice exists")
    for sli in large_slices:
        if sli[0][0] == 'SP':
            _, (s, e) = sli[0]
            if e - s > 1.0:
                sli[0][1] = (e - 1.0, e)
        if sli[-1][0] == 'SP':
            _, (s, e) = sli[-1]
            if e - s > 1.0:
                sli[-1][1] = (s, s + 1.0)
        for ph1, ph2 in zip(sli[:-1], sli[1:]):
            if ph1[1][1] != ph2[1][0]:
                raise Exception("Time mismatch")
    return large_slices
# %%
def split_to_smaller_slices(large_slice):
    cut = set()
    for idx, ph in enumerate(large_slice):
        if ph[0] == 'AP':
            cut.add(idx - 1)
            cut.add(idx)
    for idx, (ph1, ph2) in enumerate(zip(large_slice[:-1], large_slice[1:])):
        if ph1[0] == 'AP' and ph2[0] == 'SP':
            cut.discard(idx)
        elif ph1[0] == 'SP' and ph2[0] == 'AP':
            cut.discard(idx)
    small_slices = []
    tmp = []
    for idx, ph in enumerate(large_slice):
        tmp.append(ph)
        if idx in cut:
            small_slices.append(deepcopy(tmp))
            tmp = []
    if tmp:
        small_slices.append(deepcopy(tmp))
    return small_slices, [x[-1][1][1] - x[0][1][0] for x in small_slices]

#%%
def get_best_split(Ls, target):
    N = len(Ls)
    dp = [[None] * (N + 1) for _ in range(N)]
    dp_ans = [[None] * (N + 1) for _ in range(N)]
    for i in range(N):
        dp[i][i] = 0
        dp[i][i + 1] = abs(Ls[i] - 8)
        dp_ans[i][i] = -1
        dp_ans[i][i + 1] = -1
    for offset in range(2, N+1):
        for l in range(N + 1 - offset):
            r = l + offset
            best_c, best_v = -1, abs(sum(Ls[l:r]) - target)
            for c in range(l+1, r):
                v = dp[l][c] + dp[c][r]
                if v < best_v:
                    best_c = c
                    best_v = v
            dp[l][r] = best_v
            dp_ans[l][r] = best_c
    q = deque()
    q.append((0, N))
    ans = []
    while q:
        i, j = q.popleft()
        c = dp_ans[i][j]
        if c == -1:
            continue
        ans.append(c)
        q.append((i, c))
        q.append((c, j))
    ret = [0] + sorted(ans) + [len(Ls)]
    return ret, [sum(Ls[i:j]) for i, j in zip(ret[:-1], ret[1:])]
#%%
def fix_sils(slices):
    new_slices = []
    for sli in slices:
        if sli[0][0] == 'SP' and new_slices and new_slices[-1][-1][0] == 'SP' and sli[0][1][0] < new_slices[-1][-1][1][1]:
            new_slices[-1][-1][1] = (new_slices[-1][-1][1][0], sli[0][1][1])
            if len(sli) > 1:
                new_slices.append(deepcopy(sli[1:]))
        else:
            new_slices.append(deepcopy(sli))
    last_time = 0.0
    for sli in new_slices:
        for ph in sli:
            assert ph[1][0] >= last_time
            last_time = ph[1][1]
    return new_slices
#%%
def slice_wav(wav, sr, slis):
    def gen_env(s, e):
        fade_in = np.linspace(0, 1, int(0.01 * sr))
        fade_out = np.linspace(1, 0, int(0.01 * sr))
        constant = np.ones(int(e * sr) - int(s * sr) - len(fade_in) - len(fade_out))
        env = np.concatenate((fade_in, constant, fade_out))
        return env
    intvs = [(sli[0][1][0], sli[-1][1][1]) for sli in slis]
    ys = []
    for (s, e) in intvs:
        assert e - s > 0.1
        y_slice = wav[int(s * sr):int(e * sr)]
        env = gen_env(s, e)
        ys.append(y_slice * env)
    return np.concatenate(ys)
#%%
def slice_wav_command(in_wav_fn, out_wav_fn, slis):
    intvs = [(sli[0][1][0], sli[-1][1][1]) for sli in slis]
    cmds = []
    if len(intvs) == 1:
        base_time = intvs[0][0]
        length = intvs[0][1] - base_time
        cmds.append(f'ffmpeg -i "{in_wav_fn}" -ss {base_time:f} -t {length:f} -filter_complex "afade=t=in:st={base_time:f}:d=0.01,afade=t=out:st={base_time + length - Decimal("0.01"):f}:d=0.01" -c:a pcm_s24le "{out_wav_fn}" || echo ERROR && exit /b')
    else:
        for idx, (base_time, e) in enumerate(intvs):
            length = e - base_time
            cmds.append(f'ffmpeg -i "{in_wav_fn}" -ss {base_time:f} -t {length:f} -filter_complex "afade=t=in:st={base_time:f}:d=0.01,afade=t=out:st={base_time + length - Decimal("0.01"):f}:d=0.01" -c:a pcm_s24le "{out_wav_fn.with_name(out_wav_fn.stem + "_" + str(idx) + out_wav_fn.suffix)}" || echo ERROR && exit /b')
        cmds.append('sox ' + ' '.join(map(lambda x: f'"{x}"', [out_wav_fn.with_name(out_wav_fn.stem + "_" + str(idx) + out_wav_fn.suffix) for idx in range(len(intvs))])) + f' "{out_wav_fn}" || echo ERROR && exit /b')
        for idx in range(len(intvs)):
            cmds.append(f'rm "{out_wav_fn.with_name(out_wav_fn.stem + "_" + str(idx) + out_wav_fn.suffix)}"')
    return cmds
# %%
def slice_textgrid(tg, slis):
    tg_out = textgrid.TextGrid()
    intvs = [(sli[0][1][0], sli[-1][1][1]) for sli in slis]
    for tier in tg:
        tier_out = textgrid.IntervalTier(name=tier.name)
        offset = 0
        for (s, e) in intvs:
            for ph in tier:
                if ph.maxTime > s and ph.minTime < e:
                    tier_out.add(
                        round(max(ph.minTime, s) - s + offset, 7),
                        round(min(ph.maxTime, e) - s + offset, 7),
                        ph.mark
                    )
                elif ph.minTime > e:
                    break
            offset += e - s
        tg_out.append(tier_out)
    return tg_out
#%%
parser = argparse.ArgumentParser()
parser.add_argument("-iw", "--wav_folder", type=str,
                    help="specify path of data folder")
parser.add_argument("-il", "--label_folder", type=int, default=5000,
                    help="specify batch size when uploading data")
parser.add_argument("-e", "--label_ext", choices=['lab', 'TextGrid'], default='lab',
                    help="specify batch size when uploading data")
parser.add_argument("-o", "--out_folder", action='store_true',
                    help="overwrite previously written parquet files")
parser.add_argument("-br", "--br_symbol", action='store_true',
                    help="overwrite previously written parquet files")
parser.add_argument("-sil", "--sil_symbol", action='store_true',
                    help="overwrite previously written parquet files")
args = parser.parse_args()

label_path = Path(args.label_folder)
wav_path = Path(args.wav_folder)
slice_out_path = Path(args.out_folder)
#%%
cmds = []
intvs_list = []
for wav_fn in tqdm(natsorted(wav_path.glob('*.wav'))):
    fn_stem = wav_fn.stem
    label_fn = label_path / (fn_stem + ('.lab' if args.label_ext == 'lab' else '.TextGrid'))
    wav_out_dir = slice_out_path / 'wav'
    tg_out_dir = slice_out_path / 'TextGrid'
    wav_out_dir.mkdir(parents=True, exist_ok=True)
    tg_out_dir.mkdir(parents=True, exist_ok=True)

    tg, phones = read_label(label_fn, wav_fn, args.sil_symbol, args.br_symbol)
    large_slices = split_to_large_slices(phones)

    gather_slices = []
    for large_slice in large_slices:
        small_slices, _ = split_to_smaller_slices(large_slice)
        gather_slices.extend(small_slices)
    gather_slices_fixed = fix_sils(gather_slices)

    gather_Ls = [sum([ph[1][1] - ph[1][0] for ph in sli]) for sli in gather_slices_fixed]
    cut_points, lens = get_best_split(gather_Ls, 8)
    final_slices = [gather_slices_fixed[i:j] for i, j in zip(cut_points[:-1], cut_points[1:])]
    for slis in final_slices:
        for sli in slis:
            for ph1, ph2 in zip(sli[:-1], sli[1:]):
                if ph1[1][1] != ph2[1][0]:
                    raise Exception("Time mismatch")

    wav, _ = librosa.load(wav_fn, sr=44100, mono=True)
    znum = len(str(len(final_slices)))
    for num, slis in enumerate(final_slices):
        intvs_list.append((wav_fn, [(sli[0][1][0], sli[-1][1][1]) for sli in slis]))
        wav_slice = slice_wav(wav, 44100, slis)
        tg_slice = slice_textgrid(tg, slis)
        cmds.extend(slice_wav_command(wav_fn, wav_out_dir / f'{fn_stem}_{str(num).zfill(znum)}.wav', slis))
        sf.write(wav_out_dir / f'{fn_stem}_{str(num).zfill(znum)}.wav', wav_slice, 44100, subtype='PCM_24')
        tg_slice.write(tg_out_dir / f'{fn_stem}_{str(num).zfill(znum)}.TextGrid')

#%%
with open(slice_out_path / 'intervals.txt', 'w', encoding='utf-8') as f:
    for wav_fn, intvs in intvs_list:
        f.write(f'{wav_fn}\t{" ".join([",".join(map(lambda y: f"{y:.7f}", x)) for x in intvs])}\n')
with open(slice_out_path / 'slice_wav.bat', 'w', encoding='utf-8') as f:
    f.write('\n'.join(cmds))
# %%
