"""FEL timing between pump A and pump B (port of Timing/Svirgo_comparison.m,
svirgo_core.m, self_tune.m), no GUI code.

For every h5 file of a YAG timing scan, the Basler7 image stack is split into
FEL-on and FEL-off shots (I0 threshold); the relative change of the summed
counts in each ROI is one point vs the seed delay line. Pump A and pump B are
compared by cross-correlation; the lag gives the A-B delay and the suggested
pump Z position.

ROIs are [col, row, width, height], 1-based, on the image as MATLAB shows it.
One ROI is the normal case now; two ROIs were for the Wollaston prism setup.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

BASLER = '/eis-timer/Basler7/Image8'
DELAY = '/photon_source/SeedLaser/Delay_line_2'
I0_PATHS = ('/photon_diagnostics/FEL01/I0_monitor/iom_sh_a',
            '/photon_diagnostics/FEL02/I0_monitor/iom_sh_a')

# per pump trajectory 1..4, from Svirgo_comparison.m
CONVERSION_FACTOR = (0.09, 0.16, -0.86, -1.4)   # ps per mm
MAX_MOV = (65, 47, 7, 5)
MIN_MOV = (-17, -10, -2, -1.2)


def rawdata(path):
    p = os.path.normpath(path)
    return p if os.path.basename(p) == 'rawdata' else os.path.join(p, 'rawdata')


def list_h5(path):
    return sorted(glob.glob(os.path.join(rawdata(path), '*.h5')))


def read_i0(f):
    for p in I0_PATHS:
        if p in f:
            return np.asarray(f[p][()], float).ravel()
    raise KeyError('no FEL01/FEL02 iom_sh_a')


def read_stack(f):
    """Basler stack as (shot, row, col), rows/cols as in MATLAB's Image(:,:,shot)."""
    a = np.asarray(f[BASLER][()], float)
    if a.ndim == 2:
        a = a[None]
    return np.transpose(a, (0, 2, 1))


def mean_image(path):
    """Mean Basler image of the first file (for drawing the ROIs)."""
    files = list_h5(path)
    if not files:
        raise FileNotFoundError(f'no .h5 in {rawdata(path)}')
    with h5py.File(files[0], 'r') as f:
        return read_stack(f).mean(axis=0)


def file_contrast(path, fel_thr):
    """FEL-on minus FEL-off mean Basler image of one file, and its delay."""
    with h5py.File(path, 'r') as f:
        i0 = read_i0(f)
        stack = read_stack(f)
        t = float(np.mean(np.asarray(f[DELAY][()], float)))
    n = min(i0.size, stack.shape[0])
    good = i0[:n] != 0
    i0, stack = i0[:n][good], stack[:n][good]
    on = i0 > fel_thr
    if not on.any() or on.all():
        return None, t
    return stack[on].mean(axis=0) - stack[~on].mean(axis=0), t


def _smooth(img, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(np.asarray(img, float), sigma)


def contrast_image(path, fel_thr, log=print):
    """Image where the FEL-induced spot stands out: the on-off difference of the
    files with the strongest spot (top third), each with its median removed.
    Returns (image, list of (delay, spot contrast) per file)."""
    diffs, info = [], []
    for p in list_h5(path):
        try:
            d, t = file_contrast(p, fel_thr)
        except (OSError, KeyError) as e:
            log(f'skip {os.path.basename(p)}: {e}')
            continue
        if d is None:
            continue
        d = d - np.median(d)
        s = _smooth(d)
        noise = 1.4826 * np.median(np.abs(s - np.median(s))) or 1.0
        diffs.append(d)
        info.append((t, float(np.abs(s).max() / noise)))
    if not diffs:
        raise ValueError(f'no file with both FEL-on and FEL-off shots (threshold {fel_thr})')
    order = np.argsort([c for _, c in info])[::-1]
    best = order[:max(1, len(order) // 3)]
    return np.mean([diffs[k] for k in best], axis=0), info


def _edge(profile, start, step, level):
    """Walk from the peak until |profile| drops to the background level (or flips sign)."""
    k = start
    while 0 <= k + step < profile.size and profile[k + step] > level:
        k += step
    return k


def auto_roi(image, n_cuts=5, k_noise=3.0, frac=0.1, margin=1):
    """Locate the spot and pick its ROI.

    The image is smoothed, the peak is the largest |value|, its sign defines the
    spot. Along x and y, n_cuts parallel cuts through (and next to) the peak are
    followed outwards until the signal drops to the background level:
    max(k_noise * noise of the cut, frac * peak). The median edge over the cuts,
    plus a margin, gives the ROI. Returns a dict with roi (col, row, w, h; 1-based),
    the peak, the two central cuts (averaged over the n_cuts) and the limits."""
    s = _smooth(image)
    r, c = np.unravel_index(np.argmax(np.abs(s)), s.shape)
    sign = np.sign(s[r, c]) or 1.0
    s = s * sign                                  # spot positive from here on
    peak = float(s[r, c])
    # background: the image away from the peak region
    mad = 1.4826 * np.median(np.abs(s - np.median(s)))
    half = n_cuts // 2
    rows = [min(max(r + d, 0), s.shape[0] - 1) for d in range(-half, half + 1)]
    cols = [min(max(c + d, 0), s.shape[1] - 1) for d in range(-half, half + 1)]
    level = max(k_noise * mad, frac * peak)
    xl, xr, yt, yb = [], [], [], []
    for rr in rows:
        prof = s[rr]
        xl.append(_edge(prof, c, -1, level)); xr.append(_edge(prof, c, +1, level))
    for cc in cols:
        prof = s[:, cc]
        yt.append(_edge(prof, r, -1, level)); yb.append(_edge(prof, r, +1, level))
    c0 = max(int(np.median(xl)) - margin, 0); c1 = min(int(np.median(xr)) + margin, s.shape[1] - 1)
    r0 = max(int(np.median(yt)) - margin, 0); r1 = min(int(np.median(yb)) + margin, s.shape[0] - 1)
    return dict(roi=(c0 + 1, r0 + 1, c1 - c0 + 1, r1 - r0 + 1), peak=(c + 1, r + 1), sign=sign,
                amplitude=peak * sign, level=level * sign,
                cut_x=s[rows].mean(axis=0) * sign, cut_y=s[:, cols].mean(axis=1) * sign,
                x_limits=(c0 + 1, c1 + 1), y_limits=(r0 + 1, r1 + 1), n_cuts=n_cuts)


def roi_sum(stack, roi):
    c, r, w, h = (int(v) for v in roi)
    return stack[:, r - 1:r - 1 + h, c - 1:c - 1 + w].sum(axis=(1, 2))


@dataclass
class PumpTrace:
    """Accumulated points for one pump (svirgo_core output + processed list)."""
    path: str
    t: list = field(default_factory=list)
    i: list = field(default_factory=list)       # one list per ROI
    processed: list = field(default_factory=list)

    def update(self, rois, fel_thr, log=print):
        """svirgo_core.m: process the files not seen yet."""
        if not self.i:
            self.i = [[] for _ in rois]
        for p in list_h5(self.path):
            name = os.path.basename(p)
            if name in self.processed:
                continue
            try:
                with h5py.File(p, 'r') as f:
                    i0 = read_i0(f)
                    stack = read_stack(f)
                    t = float(np.mean(np.asarray(f[DELAY][()], float)))
            except (OSError, KeyError) as e:
                log(f'skip {name}: {e}')
                continue
            n = min(i0.size, stack.shape[0])
            good = i0[:n] != 0
            i0, stack = i0[:n][good], stack[:n][good]
            on = i0 > fel_thr
            off = ~on
            if not on.any() or not off.any():
                log(f'{name}: FEL on/off split empty (threshold {fel_thr})')
                self.processed.append(name)
                continue
            for k, roi in enumerate(rois):
                s = roi_sum(stack, roi)
                self.i[k].append((s[on].mean() - s[off].mean()) / s[off].mean())
            self.t.append(t)
            self.processed.append(name)
            log(f'processed {name}')

    def arrays(self):
        t = np.array(self.t)
        return t, [np.nan_to_num(np.array(v)) for v in self.i]


def auto_threshold(path):
    """self_tune.m threshold: centre of the 4th empty bin of a 100-bin I0 histogram."""
    files = list_h5(path)
    with h5py.File(files[0], 'r') as f:
        i0 = read_i0(f)
    n, edges = np.histogram(i0, 100)
    centres = (edges[:-1] + edges[1:]) / 2
    empty = np.flatnonzero(n == 0)
    return float(centres[empty[3]]) if empty.size > 3 else float(np.median(i0))


def auto_pump(path):
    """self_tune.m trajectory choice: the pump_A/<n>/Y closest to the minimum."""
    files = list_h5(path)
    vals = []
    with h5py.File(files[0], 'r') as f:
        for n in range(1, 5):
            key = f'/pump_A/{n}/Y'
            vals.append(float(np.ravel(f[key][()])[0]) if key in f else 0.0)
    return int(np.argmin(vals)) + 1


def xcorr(a, b):
    """MATLAB xcorr(a, b) for equal-length vectors: (c, lags)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(a.size, b.size)
    a = np.pad(a, (0, n - a.size))
    b = np.pad(b, (0, n - b.size))
    c = np.correlate(a, b, mode='full')
    lags = np.arange(-(n - 1), n)
    return c, lags


@dataclass
class TimingResult:
    time_diff: float
    lags: np.ndarray
    xc: np.ndarray
    current_z: float | None
    final_position: float | None
    message: str


def compare(trace_a: PumpTrace, trace_b: PumpTrace, pump: int, roi_index: int = -1):
    """The 'length(i1a)==length(i1b)' block: cross-correlate and suggest the move.

    roi_index picks the ROI curve to correlate (MATLAB used ROI 2; with one ROI
    it is ROI 1)."""
    ta, ia = trace_a.arrays()
    tb, ib = trace_b.arrays()
    xc, lags = xcorr(ia[roi_index], ib[roi_index])
    k = int(np.argmax(np.abs(xc)))
    time_diff = float(lags[k] * np.mean(np.diff(tb))) if tb.size > 1 else 0.0
    movement = time_diff / CONVERSION_FACTOR[pump - 1]
    current_z = None
    try:
        last_b = os.path.join(rawdata(trace_b.path), trace_b.processed[-1])
        with h5py.File(last_b, 'r') as f:
            current_z = float(np.ravel(f[f'/pump_A/{pump}/Z'][()])[0])
    except (OSError, KeyError, IndexError):
        pass
    msg = f'Estimated (A-B) dT {time_diff:f} (ps)'
    final = None
    if abs(time_diff) < 0.01:
        msg += '\nSearch for puntino!'
    elif current_z is None:
        msg += f'\nMove pump {pump} by {movement:f} mm (current Z not found in the files)'
    else:
        final = current_z + movement
        if final > MAX_MOV[pump - 1] or final < MIN_MOV[pump - 1]:
            msg += '\nMovement out of range, change crossing point'
        else:
            msg += f'\nEstimated final position = {final:f} (mm)'
    return TimingResult(time_diff, lags, xc, current_z, final, msg)
