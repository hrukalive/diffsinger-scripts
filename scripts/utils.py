import textgrid

def lab_to_tg_tier(lab_fn):
    tier = textgrid.IntervalTier(name='phones')
    last = None
    with open(lab_fn, 'r') as f:
        for l in f:
            if not l.strip():
                continue
            s, e, p = l.strip().split()
            if '.' not in s:
                s = int(s) / 10000000
            if '.' not in e:
                e = int(e) / 10000000
            s = round(float(s), 7)
            e = round(float(e), 7)
            if last and s != last:
                raise Exception("Time mismatch")
            last = e
            tier.add(s, e, p)
    return tier

def lab_to_tg(lab_fn):
    tg = textgrid.TextGrid()
    tg.append(lab_to_tg_tier(lab_fn))
    return tg
