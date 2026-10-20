"""Data loading for TIMER / DiProI beamline HDF5 files.

Python port of the loading layer used by AnalisiMaxiTimer200505*.m:

    MATLAB (Functions/)        Python
    -------------------        ------
    FolderStrings.m            folder_strings()
    FolderLists.m              folder_lists()
    GetFELBL.m                 get_fel_bl()
    GetEHFilters.m             get_eh_filters()
    GetImage.m                 get_image()
    GetIntensity.m             get_intensity()  (+ IntensityCache instead of the 'Old' struct)
    GetBGbyBunchNumber.m       get_bg_by_bunch_number()
    "Load Data" block of the   load_shot_file()
    main loop (nShots, delay,
    spectrometer axis)

Array orientation
-----------------
MATLAB's h5read returns datasets with their axes reversed with respect to
h5py (MATLAB is column-major). Every function here returns arrays with the
same shape and orientation as the *final* MATLAB variable, so that index
(r, c) in Python is element (r+1, c+1) in MATLAB. 1-D per-shot vectors
(MATLAB nShots x 1 columns) are returned as 1-D arrays of length nShots.

Indexing
--------
File numbers and exclude lists are 0-based in Python (MATLAB FNr = 1 is
Python index 0).
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import h5py
import numpy as np

IMAGE_PATHS_TIMER = ("/CCD/Image", "/CCD2/Image")
IMAGE_PATH_DIPROI = "/image/ccd1"
SPECTRUM_PATH = "/photon_diagnostics/Spectrometer/hor_spectrum"
DELAY_TIMER = "/padres/DelayLine/FEL_delay"
DELAY_DIPROI = ("/DPI/DelayLine", "/SLU/DelayLine")
DELAY_M2Z = "/DPI/M2Z"


def _split_path(path: str) -> list[str]:
    """Split on both '\\' and '/', so Windows paths work on any OS."""
    return re.split(r"[\\/]", path)


def _mread(f: h5py.File, path: str) -> np.ndarray:
    """Read a dataset as MATLAB's double(h5read(...)) would shape it.

    Axes reversed; 1-D data becomes an (n, 1) column, scalars (1, 1).
    """
    data = np.asarray(f[path][()], dtype=float)
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(-1, 1)
    return data.T


# --------------------------------------------------------------------------
# Folders and file lists
# --------------------------------------------------------------------------

def folder_strings(exp_folder: str, scan_nr: int, scan_str: str = "Scan_%03.0f",
                   bg_str: str = "_OnlyProbe"):
    """Port of FolderStrings.m.

    Returns (project, experiment, dataset, data_folder, bg_folder).
    """
    dataset = scan_str % scan_nr
    parts = _split_path(exp_folder)
    if parts[-1] == "":
        parts = parts[:-1]
    experiment = parts[-1]
    project = parts[-2]
    data_folder = os.path.join(exp_folder, dataset, "rawdata")
    bg_folder = os.path.join(exp_folder, dataset + bg_str, "rawdata")
    return project, experiment, dataset, data_folder, bg_folder


def _ls(folder: str, pattern: str) -> list[str]:
    """Sorted file names like MATLAB's ls on Windows (case-insensitive by name)."""
    names = [os.path.basename(p) for p in glob.glob(os.path.join(folder, pattern))]
    return sorted(names, key=lambda n: (n.upper(), n))


def folder_lists(data_folder: str, data_folder_bg: str):
    """Port of FolderLists.m.

    Returns (data_list, n_files, data_list_bg, n_files_bg). Lists hold file
    names (not full paths), sorted by name.
    """
    data_list = _ls(data_folder, "*_img.h5")
    data_list_bg = _ls(data_folder_bg, "*_img.h5")
    if not data_list and not data_list_bg:
        data_list = _ls(data_folder, "*.h5")
        data_list_bg = _ls(data_folder_bg, "*.h5")
    return data_list, len(data_list), data_list_bg, len(data_list_bg)


def _first_h5(data_folder: str) -> str:
    files = _ls(data_folder, "*.h5")
    if not files:
        raise FileNotFoundError(f"No .h5 files in {data_folder}")
    return os.path.join(data_folder, files[0])


# --------------------------------------------------------------------------
# Beamline / FEL detection
# --------------------------------------------------------------------------

def get_fel_bl(data_folder: str):
    """Port of GetFELBL.m: detect FEL01/FEL02 and TIMER/DiProI.

    Returns (fel, bl, delay_attr).
    """
    path = _first_h5(data_folder)
    with h5py.File(path, "r") as f:
        if "/photon_diagnostics/FEL01/I0_monitor/iom_sh_a" in f:
            fel = "FEL01"
        elif "/photon_diagnostics/FEL02/I0_monitor/iom_sh_a" in f:
            fel = "FEL02"
        else:
            raise KeyError(f"No FEL01/FEL02 I0 monitor in {path}")

        if DELAY_TIMER in f:
            return fel, "TIMER", DELAY_TIMER
        for attr in DELAY_DIPROI:
            if attr in f:
                return fel, "DiProI", attr
        raise KeyError(f"No known delay line dataset in {path}")


def get_eh_filters(data_folder: str):
    """Port of GetEHFilters.m. Returns (eh_filter_in, eh_filter_out) strings."""
    path = _first_h5(data_folder)
    with h5py.File(path, "r") as f:
        out = []
        for name in ("EH_filter_IN", "EH_filter_OUT"):
            v = np.asarray(f["/photon_diagnostics/Filters/" + name][()]).ravel()[0]
            out.append(v.decode() if isinstance(v, bytes) else str(v))
    return tuple(out)


# --------------------------------------------------------------------------
# Per-file readers
# --------------------------------------------------------------------------

class MissingImage(KeyError):
    """The file has no CCD image (a shot the DAQ saved without the camera)."""


def get_image(path: str, bl: str) -> np.ndarray:
    """Port of GetImage.m: CCD image as float, oriented as in MATLAB.

    MATLAB returns [] plus an error on failure; here the exception is raised.
    """
    with h5py.File(path, "r") as f:
        if bl == "DiProI":
            # rot90(h5read(...), 2)
            return np.rot90(_mread(f, IMAGE_PATH_DIPROI), 2).copy()
        for p in IMAGE_PATHS_TIMER:
            if p in f:
                # h5read(...)' == native h5py orientation
                return _mread(f, p).T.copy()
        raise MissingImage(f"No CCD image ({', '.join(IMAGE_PATHS_TIMER)}) in {path}")


def get_n_shots(path: str) -> int:
    """nShots = size(h5read(path, '/bunches'), 1)."""
    with h5py.File(path, "r") as f:
        shape = f["/bunches"].shape
    return 1 if len(shape) == 0 else int(shape[-1])


def get_delay(path: str, delay_attr: str) -> float:
    """mean(double(h5read(path, DelayAttr)))."""
    with h5py.File(path, "r") as f:
        return float(np.mean(_mread(f, delay_attr)))


@dataclass
class IntensityCache:
    """Replaces the 'Old' struct GetIntensity.m keeps in the base workspace.

    Holds the last good value of each dataset, used when a later file is
    missing it.
    """
    values: dict = field(default_factory=dict)


def get_intensity(path: str, fnr: int, start: int, n_shots: int,
                  cache: IntensityCache, *dataset_paths: str):
    """Port of GetIntensity.m.

    fnr and start are the current file index and StartEnd(1) (same base,
    either 0- or 1-based; only fnr > start matters). Returns a list with one
    array per dataset path: per-shot vectors as 1-D (n_shots,), the
    spectrometer as (n_shots, n_pixels).

    If a dataset is missing: for the first file it is zeros; for later files
    the cached value from an earlier file is reused (as MATLAB does).
    """
    out = []
    with h5py.File(path, "r") as f:
        for dpath in dataset_paths:
            name = dpath.split("/")[-1]
            if dpath in f:
                var = _mread(f, dpath)
                if name == "hor_spectrum":
                    var = var.T
                if n_shots in var.shape:
                    if var.shape[1] == n_shots and var.shape[0] != n_shots:
                        var = var.T
                    cache.values[name] = var
            elif fnr > start:
                if name not in cache.values:
                    raise KeyError(f"{dpath} missing in {path} and no earlier value cached")
                var = cache.values[name]
            elif name == "hor_spectrum":
                var = np.zeros((n_shots, 1000))
            else:
                var = np.zeros((n_shots, 1))
            out.append(var[:, 0] if var.ndim == 2 and var.shape[1] == 1 else var)
    return out


def get_spectrum_axis(path: str, n_pixels: int = 1000) -> np.ndarray:
    """Spectrometer wavelength axis (XSpect in the main script).

    Falls back to pixel numbers 1..n_pixels when the calibration datasets are
    missing, as the MATLAB try/catch does.
    """
    x = 1e-3 * (np.arange(1, n_pixels + 1) - 500)
    try:
        with h5py.File(path, "r") as f:
            wl = _mread(f, "/photon_diagnostics/Spectrometer/Wavelength")
            p2um = _mread(f, "/photon_diagnostics/Spectrometer/Pixel2micron")
            wls = _mread(f, "/photon_diagnostics/Spectrometer/WavelengthSpan")
    except KeyError:
        return np.arange(1, n_pixels + 1, dtype=float)
    xs = wl + x[np.newaxis, :] * p2um * wls
    return xs.ravel() if xs.shape[0] == 1 else xs


def get_bg_by_bunch_number(path: str, data_folder_bg: str, data_list_bg: list[str],
                           exclude_bg=()) -> str:
    """Port of GetBGbyBunchNumber.m: background file with the nearest bunch number.

    The bunch number of the data file is the token before the last '_' in
    its path (e.g. ..._123456789_img.h5). exclude_bg holds 0-based indices
    into data_list_bg.
    """
    bunch_nr = float(path.split("_")[-2])
    bunch_bg = []
    for name in data_list_bg:
        m = re.search(r"\d{7,10}", name)
        if m is None:
            raise ValueError(f"No bunch number in background file name {name}")
        bunch_bg.append(float(m.group()))
    diff = np.abs(np.array(bunch_bg) - bunch_nr)
    exclude = set(exclude_bg)

    idx = int(np.argmin(diff))
    while idx >= 0 and idx in exclude:
        idx -= 1
    if idx < 0:
        idx = int(np.argmin(diff)) + 1
    while idx in exclude:
        idx += 1
    return os.path.join(data_folder_bg, data_list_bg[idx])


# --------------------------------------------------------------------------
# One-call loader for a shot file
# --------------------------------------------------------------------------

@dataclass
class ShotFile:
    path: str
    n_shots: int
    image: np.ndarray            # CCD image (rows, cols)
    delay: float                 # mean delay-line position
    spectrum: np.ndarray         # (n_shots, n_pixels)
    iom_sh: np.ndarray           # (n_shots,)
    iom_uh: np.ndarray           # (n_shots,)
    x_spect: np.ndarray          # spectrometer axis (nm, or pixel numbers)


def delay_attr_for(bl: str, use_m2z: bool = False) -> str:
    """Delay dataset chosen in the main loop of AnalisiMaxiTimer200505.m."""
    if use_m2z:
        return DELAY_M2Z
    if bl == "DiProI":
        return DELAY_DIPROI[0]
    return DELAY_TIMER


def load_shot_file(path: str, fel: str, bl: str, fnr: int = 0, start: int = 0,
                   cache: IntensityCache | None = None,
                   delay_attr: str | None = None) -> ShotFile:
    """Everything the 'Load Data' part of the main loop reads from one file.

    delay_attr overrides the default delay dataset (the SLUpump variant uses
    '/photon_source/SeedLaser/Delay_line_2').
    """
    cache = cache if cache is not None else IntensityCache()
    n_shots = get_n_shots(path)
    image = get_image(path, bl)
    delay = get_delay(path, delay_attr or delay_attr_for(bl))
    spectrum, iom_sh, iom_uh = get_intensity(
        path, fnr, start, n_shots, cache,
        SPECTRUM_PATH,
        f"/photon_diagnostics/{fel}/I0_monitor/iom_sh_a",
        f"/photon_diagnostics/{fel}/I0_monitor/iom_uh_a",
    )
    return ShotFile(path, n_shots, image, delay, spectrum, iom_sh, iom_uh,
                    get_spectrum_axis(path))


# --------------------------------------------------------------------------
# Result text files written by the analysis (_nor.txt, _raw.txt, ...)
# --------------------------------------------------------------------------

def load_results_txt(path: str) -> np.ndarray:
    """Read a tab-separated results file (MATLAB importdata / save -ASCII)."""
    return np.loadtxt(path)
