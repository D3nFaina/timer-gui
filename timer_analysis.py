"""TG scan analysis engine (port of AnalisiMaxiTimer200505*.m), no GUI code.

One file at a time (process_file) and a whole scan (ScanAnalysis), using the
loaders in timer_io.py. The GUI (timer_gui.py) drives this module; it can
also be used from a script or the Spyder console:

    from timer_analysis import AnalysisConfig, ScanAnalysis
    cfg = AnalysisConfig(exp_folder=r'\\...\\STO', scan_nr=4)
    ana = ScanAnalysis(cfg)
    ana.run_once()          # all files, bin, write Scan_004_*.txt

What the switches do (MATLAB names in brackets)
- bg_mode: 'none' [UseBG=0], 'bunch' = nearest bunch number in the
  <Scan><bg_str> folder [UseBG=1, BG_Sig=0], 'first_n' = mean of the first
  N signal images [BG_Sig=N].
- single_image: process only file `single_image_nr` and show the ROI /
  ellipse figures, to place the ROI [UseROINr].
- loop: keep re-reading the folder and add new files [UseUpdate + while true].
- norm_mode: 'I0^3' [S.IomSh.Sum3], 'I0^2*spec' [Sum2*sum(Spect)],
  'I0*spec' [Sum*sum(Spect)] -- the variants used the three.
- x_axis: dataset used as delay; time = mean(delay) - t0 [T0Pos].

Index conventions follow the MATLAB code: ROI [col,row,width,height],
ellipse centre (BarFixXY) and the CBar/RBar/cg/rg written to the result
files are 1-based pixel coordinates, so old parameter values can be reused.
File numbers shown in the GUI are 1-based as well (FNr).

Written output (same names and columns as MATLAB, tab separated):
  <out>/<Experiment>/<Dataset>_raw.txt, _nor.txt, _pos.txt, _bg.txt, _MNF.txt,
  <Dataset>.mat (struct R) and <Dataset>_parameters.json.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

import h5py
import numpy as np
from scipy.ndimage import correlate

import timer_io

X_AXES = {
    'FEL delay (/padres/DelayLine/FEL_delay)': '/padres/DelayLine/FEL_delay',
    'Seed delay line 2 (/photon_source/SeedLaser/Delay_line_2)': '/photon_source/SeedLaser/Delay_line_2',
    'DiProI delay (/DPI/DelayLine)': '/DPI/DelayLine',
    'M2Z (/DPI/M2Z)': '/DPI/M2Z',
    'Phase shifter PS3': '/photon_source/Phase_Shifters/PS3/DeltaPhase',
    'Phase shifter PS4': '/photon_source/Phase_Shifters/PS4/DeltaPhase',
    'Phase shifter PS5': '/photon_source/Phase_Shifters/PS5/DeltaPhase',
    'Phase shifter PS6': '/photon_source/Phase_Shifters/PS6/DeltaPhase',
    'Phase shifter PS7': '/photon_source/Phase_Shifters/PS7/DeltaPhase',
}
BG_MODES = ('none', 'bunch', 'both', 'first_n')
_BUNCH_RE = re.compile(r'\d{7,10}')


def bunch_number(name):
    """Bunch number in a file name: the last group of 7-10 digits.

    GetBGbyBunchNumber.m takes the token before the last '_', which is right
    for the old ..._<bunch>_img.h5 names but gives the scan number (e.g. 2)
    for the current Scan_002_<bunch>.h5 names, so MATLAB then always picked
    the first background file. This works for both namings."""
    m = _BUNCH_RE.findall(os.path.basename(name))
    if not m:
        raise ValueError(f'no bunch number in {name}')
    return float(m[-1])


def nearest_backgrounds(path, bg_folder, bg_list, exclude_bg=()):
    """Background files sorted by distance in bunch number (exclude_bg: 1-based)."""
    if not bg_list:
        raise FileNotFoundError(f'no background files in {bg_folder}')
    b = bunch_number(path)
    order = np.argsort([abs(bunch_number(n) - b) for n in bg_list], kind='stable')
    out = [os.path.join(bg_folder, bg_list[i]) for i in order if i + 1 not in exclude_bg]
    if not out:
        raise FileNotFoundError(f'all background files in {bg_folder} are excluded')
    return out


def nearest_background(path, bg_folder, bg_list, exclude_bg=()):
    """Background file with the nearest bunch number (exclude_bg: 1-based)."""
    return nearest_backgrounds(path, bg_folder, bg_list, exclude_bg)[0]


def nearest_bg_image(path, bg_folder, bg_list, bl, exclude_bg=(), log=print):
    """Image of the nearest background file that has one. A background file
    saved without a CCD image (it happens: magnetite Scan_037) is passed
    over for the next nearest, as MATLAB's try/catch in GetImage does."""
    for p in nearest_backgrounds(path, bg_folder, bg_list, exclude_bg):
        try:
            return timer_io.get_image(p, bl)
        except timer_io.MissingImage:
            log(f'no CCD image in background {os.path.basename(p)}: using the next nearest')
    raise SkipFile(f'no background file in {bg_folder} has a CCD image')
NORM_MODES = ('I0^3', 'I0^2*spec', 'I0*spec')
BAR_MODES = ('fixed', 'max', 'barycenter')


@dataclass
class AnalysisConfig:
    exp_folder: str = ''
    scan_nr: int = 1
    scan_str: str = 'Scan_%03.0f'
    bg_str: str = '_OnlyProbe'
    out_folder: str = ''            # default: <parent of exp_folder>/results/OutPut

    bg_mode: str = 'bunch'          # 'none', 'bunch', 'both', 'first_n'
    bg_str2: str = '_NoProbe'       # second background folder for bg_mode 'both'
    bg_first_n: int = 2
    single_image: bool = False
    single_image_nr: int = 10       # 1-based, like UseROINr
    loop: bool = False
    wait_s: float = 30.0
    show_s: float = 0.3             # GUI only: minimum time each file's plots stay on screen
    fast_run: bool = False          # GUI only: True = no pause, plots drawn only when the GUI keeps up

    start_end: tuple = (1, 5000)    # 1-based, inclusive
    exclude: tuple = ()             # 1-based file numbers
    exclude_bg: tuple = ()          # 1-based background file numbers
    fel_down_thr: float = 1.0
    t0: float = 199.0
    bin: float = 0.01
    x_axis: str = 'FEL delay (/padres/DelayLine/FEL_delay)'

    roi: tuple = (420, 472, 50, 50)          # [col, row, width, height], 1-based
    ellipse: tuple = (30, 20)                # [horizontal, vertical] axes
    bar_fix: tuple = (30, 26)                # ellipse centre [x, y] in the ROI, 1-based
    bar_mode: str = 'fixed'
    lock_bar: bool = False
    bary_matlab: bool = False                # True = Barycenter.m as written (weights max - I, pulled
                                             # towards the dark pixels); only to reproduce old files
    prox_cfg: tuple = (30, 30, 30, 30)       # top, bottom, left, right

    spect_start: int = 110                   # 1-based pixel
    spect_end: int = 400
    d_spect: int = 10
    norm_mode: str = 'I0^3'
    ccd_gain: float = 36.0                   # CCD counts per detected photon (signal shot noise; 0 = off).
                                             # 36 = estimate_ccd_gain on SrO Scan_002 (1-sigma 27-47)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ('start_end', 'exclude', 'exclude_bg', 'roi', 'ellipse', 'bar_fix', 'prox_cfg'):
            if k in known:
                known[k] = tuple(known[k])
        return cls(**known)


# --------------------------------------------------------------------------
# small ports of Functions/*.m (1-based conventions kept at the interface)
# --------------------------------------------------------------------------

def mround(v):
    """MATLAB round (half away from zero)."""
    return int(np.sign(v) * np.floor(abs(v) + 0.5))


def crop_roi(image, roi):
    """CropROI.m: Image(ROI(2):ROI(2)+ROI(4)-1, ROI(1):ROI(1)+ROI(3)-1)."""
    c, r, w, h = (int(v) for v in roi)
    if r < 1 or c < 1 or r - 1 + h > image.shape[0] or c - 1 + w > image.shape[1]:
        raise IndexError(f'ROI {list(roi)} outside the image ({image.shape[1]} x {image.shape[0]})')
    return image[r - 1:r - 1 + h, c - 1:c - 1 + w]


def prox_roi_param(roi, cfg, corners=False):
    """ProxROIParam.m: the 4 proximity rectangles [top, bottom, left, right]."""
    c, r, w, h = roi
    top, bottom, left, right = cfg
    if corners:
        return [(c - left, r - top, w + left + right, top),
                (c - left, r + h, w + left + right, bottom),
                (c - left, r - top, left, h + top + bottom),
                (c + w, r - top, right, h + top + bottom)]
    return [(c, r - top, w, top),
            (c, r + h, w, bottom),
            (c - left, r, left, h),
            (c + w, r, right, h)]


def smooth3(a):
    """filter2(ones(3,3)/9, A) with zero padding."""
    return correlate(np.asarray(a, float), np.ones((3, 3)) / 9, mode='constant', cval=0.0)


def ellipse_mask(nx, ny, xc, yc, rx, ry):
    """EllipseMask.m (1-based centre)."""
    x, y = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1))
    return ((x - xc) / rx) ** 2 + ((y - yc) / ry) ** 2 <= 1


def barycenter(zoom, integer, matlab=False):
    """Centre of mass of the spot, 1-based (row, col).

    Barycenter.m weights the 3x3-smoothed image by max(I) - I, i.e. the dark
    pixels weigh most and the 'barycentre' is pulled away from the spot
    (matlab=True keeps that, to reproduce old _pos files). Fixed version:
    weights are the smoothed intensity above half way between the median
    (background) and the maximum, so only the spot counts and noise far from
    it does not drag the centre to the middle of the image."""
    img = smooth3(zoom)
    if matlab:
        img = img - img.max()
        img = img / img.sum()
    else:
        med, top = float(np.median(img)), float(img.max())
        img = np.clip(img - (med + 0.5 * (top - med)), 0.0, None)
        if not img.sum() > 0:   # flat image: centre of the ROI
            img = np.ones_like(img)
        img = img / img.sum()
    m, n = img.shape
    i, j = np.meshgrid(np.arange(1, m + 1), np.arange(1, n + 1), indexing='ij')
    rb = float(np.sum(i * img))
    cb = float(np.sum(j * img))
    if integer:
        return mround(rb), mround(cb)
    return rb, cb


def max_position(zoom_s):
    """[ValM,RowM]=max(Z); [~,ColM]=max(ValM) -> 1-based (row, col)."""
    row_m = np.argmax(zoom_s, axis=0)
    col = int(np.argmax(zoom_s[row_m, np.arange(zoom_s.shape[1])]))
    return int(row_m[col]) + 1, col + 1


def shot_stats(x, good):
    """Stats_kk_Update.m for one variable (std with N-1, Sum3 divided by #good)."""
    v = np.asarray(x, float)[good]
    ng = int(np.sum(good))
    std = lambda a: float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return dict(Sum=float(v.sum()), Ave=float(v.mean()), Std=std(v),
                Sum2=float((v ** 2).sum()), Ave2=float((v ** 2).mean()), Std2=std(v ** 2),
                Sum3=float((v ** 3).sum()) / ng, Ave3=float((v ** 3).mean()), Std3=std(v ** 3))


def spectrum_baseline(spectrum, start, end, d):
    """Spectrum ROI and Sum block: linear baseline between the two edge windows.

    start/end/d are 1-based like StartSpect/EndSpect/DSpect. Returns
    (Spect [nShots x (end-start+1)], IomSpect [nShots])."""
    s = np.asarray(spectrum, float)
    c1 = s[:, start - 1:start + d].mean(axis=1)
    c2 = s[:, end - d - 1:end].mean(axis=1)
    a = (c2 - c1) / (end - start)
    ramp = np.arange(1, end - start + 2)
    spect = s[:, start - 1:end] - (c1[:, None] + a[:, None] * ramp[None, :])
    return spect, spect.sum(axis=1)


def results_bin(bin_size, time, **vars_):
    """Results_Bin.m: group by round(Time/Bin), [mean std] per group.

    Groups come out in order of first appearance, as in MATLAB."""
    time = np.asarray(time, float)
    keys = np.array([mround(t / bin_size) for t in time])
    out = {'Time': []}
    for k in vars_:
        out[k] = []
    seen = []
    for k in keys:
        if k in seen:
            continue
        seen.append(k)
        sel = keys == k
        for name, arr in [('Time', time)] + [(n, np.asarray(v, float)) for n, v in vars_.items()]:
            a = arr[sel]
            out[name].append([a.mean(), float(np.std(a, ddof=1)) if a.size > 1 else 0.0])
    return {k: np.array(v).reshape(-1, 2) for k, v in out.items()}


def results_bin_err(bin_size, time, err):
    """Propagated error of each Results_Bin group mean: sqrt(sum(err^2)) / n.
    Same group order as results_bin. Returns (err, n) per group."""
    keys = np.array([mround(t / bin_size) for t in np.asarray(time, float)])
    err = np.asarray(err, float)
    out, seen = [], []
    for k in keys:
        if k in seen:
            continue
        seen.append(k)
        e = err[keys == k]
        out.append((np.sqrt(np.sum(e ** 2)) / e.size, e.size))
    out = np.array(out, float).reshape(-1, 2)
    return out[:, 0], out[:, 1].astype(int)


def pixel_noise(image_sub, prox):
    """Per-pixel noise of the background-subtracted image: pooled std of the
    proximity frames, each frame around its own mean (so gradients between
    frames do not count as noise). Returns (sigma_px, sigma of prbg)."""
    ss, dof, var_prbg = 0.0, 0, 0.0
    for p in prox:
        a = crop_roi(image_sub, p).astype(float).ravel()
        if a.size < 2:
            continue
        ss += float(((a - a.mean()) ** 2).sum())
        dof += a.size - 1
    sigma = np.sqrt(ss / dof) if dof > 0 else 0.0
    # prbg = sum over frames of mean(frame) / 4
    for p in prox:
        if p[2] * p[3] > 0:
            var_prbg += sigma ** 2 / (p[2] * p[3]) / 16
    return sigma, np.sqrt(var_prbg)


def norm_rel_error(norm_mode, i0, spec):
    """Relative standard error of the normalisation factor, from the shot-to-shot
    spread of the good shots (delta method, I0-spectrometer covariance included).
    I0^3 -> mean(I0^3); I0^2*spec -> mean(I0^2)*mean(S); I0*spec -> mean(I0)*mean(S)."""
    i0 = np.asarray(i0, float)
    spec = np.asarray(spec, float)
    n = i0.size
    if n < 2:
        return 0.0
    if norm_mode == 'I0^3':
        x = i0 ** 3
        return float(np.std(x, ddof=1) / (abs(x.mean()) * np.sqrt(n)))
    x = i0 ** 2 if norm_mode == 'I0^2*spec' else i0
    c = np.cov(x, spec, ddof=1)
    mx, my = x.mean(), spec.mean()
    rel2 = (c[0, 0] / mx ** 2 + c[1, 1] / my ** 2 + 2 * c[0, 1] / (mx * my)) / n
    return float(np.sqrt(max(rel2, 0.0)))


def estimate_ccd_gain(results, t_min=3.0):
    """Counts per photon from the data: the scatter of each point around a local
    quadratic through its 4 neighbours is fitted (maximum likelihood) with
    var = (pixel + normalisation error)^2 + g * TG counts / (norm factor * shots)^2.
    Any noise that grows like the signal ends up in g, so it is an effective value.
    Returns (g, g_low, g_high, n_points), the range being 1 sigma."""
    from scipy.optimize import minimize_scalar
    rs = sorted(results, key=lambda r: r.time)
    t = np.array([r.time for r in rs], float)
    y = np.array([r.tg_sum_nor for r in rs], float)
    k = np.array([1 / abs(r.mean_norm_factor * r.n_good) for r in rs])
    var0 = np.array([r.tg_nor_err for r in rs], float) ** 2
    if np.isfinite(var0).all() and len(rs) > 0:
        # tg_nor_err may already hold a shot-noise term: remove it
        g_used = getattr(rs[0], 'gain_used', 0.0)
        var0 = var0 - g_used * np.array([max(r.tg_sum, 0) for r in rs]) * k ** 2
    vs = np.array([max(r.tg_sum, 0) for r in rs]) * k ** 2
    terms = []
    for i in range(2, len(t) - 2):
        if t[i] < t_min:
            continue
        j = np.array([i - 2, i - 1, i + 1, i + 2])
        A = np.vander(t[j] - t[i], 3)
        h = np.linalg.pinv(A)[-1]
        terms.append((y[i] - h @ y[j], var0[i] + np.sum(h ** 2 * var0[j]), vs[i] + np.sum(h ** 2 * vs[j])))
    if len(terms) < 10:
        raise ValueError(f'only {len(terms)} points after t = {t_min}: too few to estimate the gain')
    r, a, b = (np.array(v) for v in zip(*terms))

    def nll(lg):
        V = a + np.exp(lg) * b
        return 0.5 * np.sum(np.log(V) + r ** 2 / V)
    best = minimize_scalar(nll, bounds=(np.log(0.01), np.log(1e5)), method='bounded')
    grid = np.linspace(np.log(0.01), np.log(1e5), 800)
    ok = grid[np.array([nll(v) for v in grid]) < best.fun + 0.5]
    return float(np.exp(best.x)), float(np.exp(ok.min())), float(np.exp(ok.max())), len(terms)


# --------------------------------------------------------------------------
# per-file processing
# --------------------------------------------------------------------------

@dataclass
class FileResult:
    fnr: int                      # 1-based file number
    path: str
    time: float
    n_shots: int
    n_good: int
    good: np.ndarray
    iom_sh: np.ndarray
    spect_ave: np.ndarray
    x_spect: np.ndarray
    image_sub: np.ndarray
    zoom: np.ndarray
    cbar: float
    rbar: float
    mean_norm_factor: float
    iom_sh_ave: float
    iom_sh_ave2: float
    tg_sum: float = np.nan
    tg_sum_nor: float = np.nan
    flux_fac_sum: float = np.nan
    cg: float = np.nan
    rg: float = np.nan
    ell_mask: np.ndarray | None = None
    spect_int: np.ndarray | None = None      # per-shot spectrometer integral (normalisation window)
    spect_full: np.ndarray | None = None     # mean raw spectrum of the good shots, all pixels
    prbg: float = np.nan                     # proximity background per pixel
    bg_zoom: np.ndarray | None = None        # ROI of the background image subtracted from this file
    tg_err: float = np.nan                   # 1-sigma error of tg_sum (CCD counts)
    tg_nor_err: float = np.nan               # 1-sigma error of tg_sum_nor (CCD error + normalisation error)
    px_noise: float = np.nan                 # per-pixel noise from the proximity frames (counts)
    norm_rel_err: float = np.nan             # relative error of the normalisation factor
    gain_used: float = 0.0                   # ccd_gain used for tg_err


class SkipFile(Exception):
    pass


def read_x_axis(path, dataset):
    with h5py.File(path, 'r') as f:
        if dataset not in f:
            raise KeyError(f'{dataset} not in {os.path.basename(path)}')
        return float(np.mean(np.asarray(f[dataset][()], float)))


class ScanAnalysis:
    """State of one scan analysis (replaces the MATLAB base workspace)."""

    def __init__(self, cfg: AnalysisConfig, log=print):
        self.cfg = cfg
        self.log = log
        (self.project, self.experiment, self.dataset,
         self.data_folder, self.bg_folder) = timer_io.folder_strings(
            cfg.exp_folder, cfg.scan_nr, cfg.scan_str, cfg.bg_str)
        self.fel, self.bl, _ = timer_io.get_fel_bl(self.data_folder)
        self.cache = timer_io.IntensityCache()
        self.results: dict[int, FileResult] = {}
        self.lock_ref: tuple | None = None
        self.binned = None
        self.bg_check = None
        self.bg_check_zoom = None
        self._bg_first_n_img = None
        self.refresh_lists()

    # ---- folders -------------------------------------------------------
    def refresh_lists(self):
        (self.data_list, self.n_files,
         self.data_list_bg, self.n_files_bg) = timer_io.folder_lists(self.data_folder, self.bg_folder)

    @property
    def out_dir(self):
        base = self.cfg.out_folder or os.path.join(os.path.dirname(os.path.normpath(self.cfg.exp_folder)),
                                                   'results', 'OutPut')
        return os.path.join(base, self.experiment)

    def file_path(self, fnr):
        return os.path.join(self.data_folder, self.data_list[fnr - 1])

    # ---- background ----------------------------------------------------
    def background(self, path, image):
        cfg = self.cfg
        if cfg.bg_mode == 'none':
            return np.zeros_like(image)
        if cfg.bg_mode == 'first_n':
            if self._bg_first_n_img is None:
                # the first N files that have a CCD image (an image-less file is passed over)
                acc, n = np.zeros_like(image), 0
                for ll in range(1, self.n_files + 1):
                    if n == cfg.bg_first_n:
                        break
                    try:
                        acc += timer_io.get_image(self.file_path(ll), self.bl)
                        n += 1
                    except timer_io.MissingImage:
                        self.log(f'no CCD image in file {ll}: not used for the first-N background')
                if n == 0:
                    raise SkipFile('none of the first files has a CCD image')
                self._bg_first_n_img = acc / n
            return self._bg_first_n_img
        if self.n_files_bg == 0:
            raise SkipFile(f'no background files in {self.bg_folder} (choose another background mode)')
        bg = nearest_bg_image(path, self.bg_folder, self.data_list_bg, self.bl, cfg.exclude_bg, self.log)
        if cfg.bg_mode == 'both':
            # signal - NoProbe - OnlyProbe: the camera offset is then removed twice;
            # that constant is taken out again by the proximity background.
            folder2 = os.path.join(os.path.dirname(os.path.dirname(self.data_folder)),
                                   self.dataset + cfg.bg_str2, 'rawdata')
            list2 = timer_io.folder_lists(folder2, folder2)[0]
            if not list2:
                raise SkipFile(f'no background files in {folder2}')
            bg = bg + nearest_bg_image(path, folder2, list2, self.bl, log=self.log)
        return bg

    # ---- one file ------------------------------------------------------
    def process_file(self, fnr, roi_only=False) -> FileResult:
        cfg = self.cfg
        path = self.file_path(fnr)
        n_shots = timer_io.get_n_shots(path)
        if n_shots <= 1:
            raise SkipFile(f'file {fnr}: {n_shots} shot(s)')
        try:
            image = timer_io.get_image(path, self.bl)
        except timer_io.MissingImage:
            raise SkipFile(f'file {fnr} ({os.path.basename(path)}) has no CCD image') from None
        bg_image = self.background(path, image)
        image_sub = image - bg_image

        delay = read_x_axis(path, X_AXES[cfg.x_axis])
        time = delay if cfg.x_axis.startswith('M2Z') else delay - cfg.t0

        spectrum, iom_sh, iom_uh = timer_io.get_intensity(
            path, fnr, cfg.start_end[0], n_shots, self.cache,
            timer_io.SPECTRUM_PATH,
            f'/photon_diagnostics/{self.fel}/I0_monitor/iom_sh_a',
            f'/photon_diagnostics/{self.fel}/I0_monitor/iom_uh_a')
        x_spect = timer_io.get_spectrum_axis(path)
        spect, iom_spect = spectrum_baseline(spectrum, cfg.spect_start, cfg.spect_end, cfg.d_spect)

        good = (iom_spect != 0) & (np.asarray(iom_sh) > cfg.fel_down_thr)
        if not good.any():
            if not np.any(iom_spect != 0):
                raise SkipFile(f'file {fnr}: the spectrometer signal is zero in pixels {cfg.spect_start}-{cfg.spect_end} '
                               f'for every shot; set "Spectrum start, end" (ROI / spectrum tab) where the spectrum is')
            raise SkipFile(f'file {fnr}: no good shots (I0 below {cfg.fel_down_thr})')
        n_good = int(good.sum())
        spect_ave = spect[good].mean(axis=0)
        st = shot_stats(iom_sh, good)
        if cfg.norm_mode == 'I0^3':
            mnf = st['Sum3']
        elif cfg.norm_mode == 'I0^2*spec':
            mnf = st['Sum2'] * spect_ave.sum()
        else:
            mnf = st['Sum'] * spect_ave.sum()

        prox = prox_roi_param(cfg.roi, cfg.prox_cfg)
        extra = dict(spect_int=np.asarray(iom_spect, float), spect_full=np.asarray(spectrum, float)[good].mean(axis=0),
                     bg_zoom=crop_roi(np.asarray(bg_image, float), cfg.roi))
        if roi_only:
            zoom = crop_roi(image_sub, cfg.roi)
            r, c = self._bar_position(zoom, smooth3(zoom), time, roi_mode=True)
            return FileResult(fnr, path, time, n_shots, n_good, good, np.asarray(iom_sh), spect_ave,
                              x_spect, image_sub, zoom, c, r, mnf, st['Ave'], st['Ave2'], **extra)

        prbg = sum(crop_roi(image_sub, p).sum() / (4 * p[2] * p[3]) for p in prox)
        zoom = crop_roi(image_sub, cfg.roi) - prbg
        rbar, cbar = self._bar_position(zoom, smooth3(zoom), time)

        ex, ey = cfg.ellipse
        mask = ellipse_mask(zoom.shape[1], zoom.shape[0], cbar, rbar, ex / 2, ey / 2)
        r0, r1 = rbar - mround(ey / 2), rbar + mround(ey / 2)
        c0, c1 = cbar - mround(ex / 2), cbar + mround(ex / 2)
        if r0 < 1 or c0 < 1 or r1 > zoom.shape[0] or c1 > zoom.shape[1]:
            raise SkipFile(f'file {fnr}: ellipse at ({cbar}, {rbar}) does not fit in the ROI')
        zoom_ell = zoom[r0 - 1:r1, c0 - 1:c1]
        tg_sum = float((zoom * mask).sum())
        rg, cg = barycenter(zoom_ell, 0, self.cfg.bary_matlab)

        # errors: pixel noise in the ellipse + error of the subtracted prbg
        # + shot noise of the signal; then the normalisation factor's error
        n_ell = float(mask.sum())
        sig_px, sig_prbg = pixel_noise(image_sub, prox)
        tg_var = n_ell * sig_px ** 2 + (n_ell * sig_prbg) ** 2 + cfg.ccd_gain * max(tg_sum, 0.0)
        tg_err = float(np.sqrt(tg_var))
        rel_n = norm_rel_error(cfg.norm_mode, np.asarray(iom_sh, float)[good], np.asarray(iom_spect, float)[good])
        tg_nor = tg_sum / (mnf * n_good)
        tg_nor_err = float(np.hypot(tg_err / abs(mnf * n_good), tg_nor * rel_n))
        return FileResult(fnr, path, time, n_shots, n_good, good, np.asarray(iom_sh), spect_ave,
                          x_spect, image_sub, zoom, cbar, rbar, mnf, st['Ave'], st['Ave2'],
                          tg_sum=tg_sum, tg_sum_nor=tg_nor, tg_err=tg_err, tg_nor_err=tg_nor_err,
                          px_noise=float(sig_px), norm_rel_err=rel_n, gain_used=cfg.ccd_gain,
                          flux_fac_sum=float(zoom_ell.sum()), cg=cg, rg=rg, ell_mask=mask,
                          prbg=float(prbg), **extra)

    def _bar_position(self, zoom, zoom_s, time, roi_mode=False):
        """Ellipse centre (1-based row, col), same rules as the MATLAB loop."""
        cfg = self.cfg
        if cfg.bar_mode == 'fixed' or (not roi_mode and time < -0.35):
            return int(cfg.bar_fix[1]), int(cfg.bar_fix[0])
        if cfg.bar_mode == 'max':
            return max_position(zoom_s)
        if not roi_mode and cfg.lock_bar and self.results and time > 0.3:
            if self.lock_ref is None:
                last = self.results[max(self.results)]
                self.lock_ref = (int(last.rbar), int(last.cbar))
            return self.lock_ref
        return barycenter(zoom, 1, self.cfg.bary_matlab)

    # ---- whole scan ----------------------------------------------------
    def files_to_process(self):
        """File numbers still to do. In loop mode the last 3 analysed files are
        redone, as MATLAB's UseUpdate step back does."""
        cfg = self.cfg
        last = min(cfg.start_end[1], self.n_files)
        done = sorted(self.results)
        first = cfg.start_end[0]
        if done:
            first = max(first, done[max(0, len(done) - 3)])
        return [f for f in range(first, last + 1) if f not in cfg.exclude]

    def process_files(self, fnrs, on_file=None, stop=None):
        for fnr in fnrs:
            if stop is not None and stop():
                return
            try:
                res = self.process_file(fnr)
            except SkipFile as e:
                self.log(f'skip: {e}')
                continue
            except (OSError, KeyError) as e:
                self.log(f'skip file {fnr}: {e}')
                continue
            self.results[fnr] = res
            if on_file:
                on_file(res)

    def bin_results(self):
        rs = [self.results[k] for k in sorted(self.results)]
        if not rs:
            return None
        time = [r.time for r in rs]
        R = results_bin(self.cfg.bin, time,
                        TGSum=[r.tg_sum for r in rs], TGSumNor=[r.tg_sum_nor for r in rs],
                        FluxFacSum=[r.flux_fac_sum for r in rs], CBar=[r.cbar for r in rs],
                        RBar=[r.rbar for r in rs], cg=[r.cg for r in rs], rg=[r.rg for r in rs],
                        IomSh_Ave=[r.iom_sh_ave for r in rs], IomSh_Ave2=[r.iom_sh_ave2 for r in rs])
        mnf = np.array([r.mean_norm_factor for r in rs])
        R['TGSignal'] = R['TGSumNor'] * mnf.mean()
        # propagated CCD + normalisation error of each bin (the MATLAB std column is the scatter)
        err, n = results_bin_err(self.cfg.bin, time, [r.tg_nor_err for r in rs])
        R['TGSignalErr'] = err * mnf.mean()
        R['NFiles'] = n
        R['MeanNormFactor'] = float(mnf.mean())
        self.binned = R
        return R

    def check_background(self, on_file=None):
        """Check BackGround Stability block: ROI*ellipse sum of each BG image /
        mean(I0), relative to the first one."""
        if self.n_files_bg == 0 or not self.results:
            self.bg_check = None
            return None
        mask = self.results[max(self.results)].ell_mask
        ref = None
        sums, delays = [], []
        for i, name in enumerate(self.data_list_bg, start=1):
            if i in self.cfg.exclude_bg:
                continue
            p = os.path.join(self.bg_folder, name)
            try:
                img = timer_io.get_image(p, self.bl)
                with h5py.File(p, 'r') as f:
                    i0 = np.mean(np.asarray(f[f'/photon_diagnostics/{self.fel}/I0_monitor/iom_sh_a'][()], float))
                img = img / i0
                if ref is None:
                    ref = img
                else:
                    img = img - ref
                zoom = crop_roi(img, self.cfg.roi)
                self.bg_check_zoom = zoom
                sums.append(float((zoom * mask).sum()))
                delays.append(timer_io.get_delay(p, timer_io.delay_attr_for(self.bl)))
                if on_file:
                    on_file(zoom)
            except (OSError, KeyError, IndexError) as e:
                self.log(f'skip background {name}: {e}')
        if sums:
            sums[0] = 0.0
        self.bg_check = (np.array(delays), np.array(sums))
        return self.bg_check

    def write_outputs(self):
        R = self.binned if self.binned is not None else self.bin_results()
        if R is None:
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        fp = os.path.join(self.out_dir, self.dataset)
        fmt = '%.7e'
        raw = np.hstack([R['Time'], R['TGSum'], R['IomSh_Ave'], R['IomSh_Ave2'],
                         R['FluxFacSum'], R['CBar'], R['RBar']])
        np.savetxt(fp + '_raw.txt', raw, fmt=fmt, delimiter='\t')
        np.savetxt(fp + '_nor.txt', np.hstack([R['Time'][:, :1], R['TGSignal']]), fmt=fmt, delimiter='\t')
        np.savetxt(fp + '_pos.txt', np.column_stack([R['Time'][:, 0], R['cg'][:, 0], R['rg'][:, 0],
                                                     R['CBar'][:, 0], R['RBar'][:, 0]]),
                   fmt=fmt, delimiter='\t')
        if self.bg_check is not None and len(self.bg_check[0]):
            np.savetxt(fp + '_bg.txt', np.column_stack(self.bg_check), fmt=fmt, delimiter='\t')
        np.savetxt(fp + '_MNF.txt', [R['MeanNormFactor']], fmt=fmt, delimiter='\t')
        np.savetxt(fp + '_err.txt', np.column_stack([R['Time'][:, 0], R['TGSignal'][:, 0], R['TGSignalErr'],
                                                     R['TGSignal'][:, 1], R['NFiles']]),
                   fmt=fmt, delimiter='\t',
                   header='time\tTGSignal\terror (CCD counts + normalisation, propagated)\tstd between files in bin\tfiles in bin')
        try:
            from scipy.io import savemat
            savemat(fp + '.mat', {'R': {k: v for k, v in R.items()}})
        except Exception as e:  # .mat is a convenience copy, txt files are the result
            self.log(f'.mat not written: {e}')
        self.cfg.save(fp + '_parameters.json')
        return fp

    def run_once(self, on_file=None, stop=None):
        self.refresh_lists()
        self.process_files(self.files_to_process(), on_file=on_file, stop=stop)
        self.bin_results()
        self.check_background()
        return self.write_outputs()
