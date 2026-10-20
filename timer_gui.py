"""TIMER analysis GUI.

Start it from Anaconda -> Spyder: open this file and press Run (F5), or in
the console:   %run timer_gui.py
(If Spyder asks, "Run in dedicated console" is the safest choice. The window
blocks the console until it is closed.)

Main window: TG scan analysis (port of AnalisiMaxiTimer200505*.m)
  - Background box: subtract or not, and from where.
  - Single image box: analyse one file only to place the ROI / ellipse.
  - Loop box: keep re-reading the scan folder and add the new points.
Buttons open:
  - Timing (Svirgo): pump A vs pump B timing with 1 or 2 ROIs.
  - Trace analysis: thermal fit, FFT of the residual, global fits with
    1 / 2 / 3 frequencies (Den's notebook models).

Needs numpy, scipy, matplotlib, h5py (all in Anaconda); tabulate optional.
"""

from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import RectangleSelector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timer_analysis as ta  # noqa: E402
import timer_timing as tt  # noqa: E402
import trace_fit as tf  # noqa: E402

SETTINGS_FILE = os.path.join(os.path.expanduser('~'), '.timer_gui_settings.json')

# --------------------------------------------------------------------------
# look and feel
# --------------------------------------------------------------------------

PALETTE = dict(
    bg='#eef2ff',          # window background (indigo 50)
    panel='#ffffff',
    accent='#4f46e5',      # indigo
    accent_dark='#312e81',
    accent_soft='#c7d2fe',
    text='#1e293b',
    muted='#64748b',
    run='#16a34a', stop='#f59e0b', abort='#dc2626', tool='#0891b2',
    console_bg='#0f172a', console_fg='#a5f3fc',
)
PLOT_COLORS = ['#4f46e5', '#f97316', '#10b981', '#e11d48', '#0ea5e9', '#a855f7', '#eab308', '#14b8a6']
STATUS_COLORS = dict(idle='#e0e7ff', busy='#dbeafe', ok='#dcfce7', warn='#fef3c7', bad='#fee2e2')


def apply_style(root):
    """ttk theme + matplotlib style used by every window."""
    import matplotlib as mpl
    from cycler import cycler
    P = PALETTE
    st = ttk.Style(root)
    try:
        st.theme_use('clam')
    except tk.TclError:
        pass
    base_font = ('Helvetica', 10)
    root.configure(background=P['bg'])
    st.configure('.', background=P['bg'], foreground=P['text'], font=base_font,
                 fieldbackground=P['panel'], bordercolor=P['accent_soft'], lightcolor=P['bg'],
                 darkcolor=P['accent_soft'], troughcolor=P['accent_soft'], focuscolor=P['accent'])
    st.configure('TFrame', background=P['bg'])
    st.configure('Card.TFrame', background=P['panel'])
    st.configure('TLabel', background=P['bg'])
    st.configure('TLabelframe', background=P['bg'], bordercolor=P['accent_soft'], relief='solid')
    st.configure('TLabelframe.Label', background=P['bg'], foreground=P['accent'],
                 font=('Helvetica', 10, 'bold'))
    st.configure('TCheckbutton', background=P['bg'])
    st.configure('Head.TCheckbutton', background=P['bg'], foreground=P['accent_dark'],
                 font=('Helvetica', 10, 'bold'))
    st.configure('TRadiobutton', background=P['bg'])
    st.map('TCheckbutton', background=[('active', P['bg'])], indicatorcolor=[('selected', P['accent'])])
    st.map('TRadiobutton', background=[('active', P['bg'])], indicatorcolor=[('selected', P['accent'])])
    st.configure('TEntry', fieldbackground=P['panel'], bordercolor=P['accent_soft'], padding=2)
    st.map('TEntry', bordercolor=[('focus', P['accent'])], fieldbackground=[('disabled', '#f1f5f9')])
    st.configure('TCombobox', fieldbackground=P['panel'], arrowcolor=P['accent'])
    st.configure('TNotebook', background=P['bg'], borderwidth=0, tabmargins=(2, 4, 2, 0))
    st.configure('TNotebook.Tab', background=P['accent_soft'], foreground=P['accent_dark'],
                 padding=(12, 5), font=('Helvetica', 10, 'bold'), borderwidth=0)
    st.map('TNotebook.Tab', background=[('selected', P['accent']), ('active', '#a5b4fc')],
           foreground=[('selected', 'white')])
    st.configure('TButton', background=P['accent_soft'], foreground=P['accent_dark'], padding=(8, 5),
                 borderwidth=0, font=('Helvetica', 10, 'bold'))
    st.map('TButton', background=[('disabled', '#e2e8f0'), ('active', '#a5b4fc')],
           foreground=[('disabled', '#94a3b8')])
    for name, col in (('Run', P['run']), ('Stop', P['stop']), ('Abort', P['abort']),
                      ('Tool', P['tool']), ('Accent', P['accent'])):
        st.configure(f'{name}.TButton', background=col, foreground='white')
        st.map(f'{name}.TButton', background=[('disabled', '#e2e8f0'), ('active', col)],
               foreground=[('disabled', '#94a3b8')])
    mpl.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': '#fbfcff', 'axes.edgecolor': '#cbd5e1',
        'axes.grid': True, 'grid.color': '#e2e8f0', 'grid.linewidth': 0.7,
        'axes.titleweight': 'bold', 'axes.titlecolor': P['accent_dark'], 'axes.titlesize': 10,
        'axes.labelcolor': '#334155', 'xtick.color': '#475569', 'ytick.color': '#475569',
        'axes.prop_cycle': cycler(color=PLOT_COLORS), 'lines.linewidth': 1.6,
        'legend.frameon': True, 'legend.framealpha': 0.85, 'legend.edgecolor': '#e2e8f0',
        'font.size': 9, 'image.cmap': 'jet',
    })


def header(parent, title, subtitle=''):
    """Coloured title bar at the top of a window."""
    P = PALETTE
    bar = tk.Frame(parent, background=P['accent_dark'])
    bar.pack(side='top', fill='x')
    tk.Frame(bar, background=P['accent'], height=4).pack(side='bottom', fill='x')
    tk.Label(bar, text=title, background=P['accent_dark'], foreground='white',
             font=('Helvetica', 15, 'bold'), padx=12, pady=6).pack(side='left')
    if subtitle:
        tk.Label(bar, text=subtitle, background=P['accent_dark'], foreground=P['accent_soft'],
                 font=('Helvetica', 10), padx=4).pack(side='left')
    return bar


def console(parent, height=10, width=48):
    return tk.Text(parent, height=height, width=width, font=('Courier', 9), background=PALETTE['console_bg'],
                   foreground=PALETTE['console_fg'], insertbackground='white', relief='flat',
                   padx=6, pady=4)


# --------------------------------------------------------------------------
# small widgets
# --------------------------------------------------------------------------

def parse_list(text, cast=float):
    """'1, 2, 5:7' -> [1, 2, 5, 6, 7] (ranges inclusive, like MATLAB)."""
    out = []
    for tok in text.replace(';', ',').replace(' ', ',').split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            a, b = tok.split(':')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(cast(tok))
    return out


def fmt_list(v):
    return ', '.join(f'{x:g}' if isinstance(x, float) else str(x) for x in v)




# display-only contrast enhancement of the CCD images (the analysis uses the raw data)
STRETCH_SMOOTH = 1.5           # gaussian smoothing, pixels
STRETCH_RANGE = (50, 99.95)    # colour range: percentiles of the smoothed image


STRETCH_MAX_PX = 512           # bigger images are block-averaged for display (faster redraw)


def stretch(image):
    """Display copy of a CCD image: block-averaged down to <= STRETCH_MAX_PX,
    lightly smoothed; colour range from the median (background, blue) to the
    99.95th percentile (peak, red), so the TG spot stands out."""
    from scipy.ndimage import gaussian_filter
    img = np.nan_to_num(np.asarray(image, float))
    b = int(np.ceil(max(img.shape) / STRETCH_MAX_PX))
    if b > 1:
        h, w = img.shape[0] // b * b, img.shape[1] // b * b
        img = img[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
    img = gaussian_filter(img, STRETCH_SMOOTH / b)
    lo, hi = np.percentile(img, STRETCH_RANGE)
    return img, lo, (hi if hi > lo else lo + 1)


def show_stretched(ax, image, colorbar=True):
    """Linear, contrast-enhanced image (see stretch) in pixel coordinates of the
    original image; the colorbar is in CCD counts per pixel."""
    img, lo, hi = stretch(image)
    ny, nx = np.shape(image)
    im = ax.imshow(img, cmap='jet', vmin=lo, vmax=hi, origin='upper', interpolation='nearest',
                   extent=(0.5, nx + 0.5, ny + 0.5, 0.5))
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='counts / pixel')
    return im

def error_band(ax, x, y, err, color, marker='o', ms=3, label=None):
    """Line with markers and a fill_between band of +-err, drawn in time order.
    Returns the handle for set_error_band."""
    x, y, err = (np.asarray(v, float) for v in (x, y, err))
    o = np.argsort(x, kind='stable')
    x, y, err = x[o], y[o], err[o]
    band = ax.fill_between(x, y - err, y + err, color=color, alpha=0.25, lw=0)
    line, = ax.plot(x, y, '-', marker=marker, ms=ms, color=color, label=label)
    return {'ax': ax, 'band': band, 'line': line, 'color': color}


def set_error_band(h, x, y, err):
    """New data for a band drawn by error_band."""
    x, y, err = (np.asarray(v, float) for v in (x, y, err))
    o = np.argsort(x, kind='stable')
    x, y, err = x[o], y[o], err[o]
    h['line'].set_data(x, y)
    h['lim'] = np.column_stack([np.r_[x, x], np.r_[y - err, y + err]])   # for the axis limits
    h['band'].remove()
    h['band'] = h['ax'].fill_between(x, y - err, y + err, color=h['color'], alpha=0.25, lw=0)


def add_data_band(figs, x, y, err, color='#e11d48'):
    """Shade +-err around every line of the figures that draws the data (x, y)."""
    if err is None:
        return
    x, y, err = (np.asarray(v, float) for v in (x, y, err))
    for fig in figs:
        for ax in fig.axes:
            for ln in list(ax.get_lines()):
                lx, ly = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
                if lx.shape == x.shape and np.allclose(lx, x) and np.allclose(ly, y):
                    ax.fill_between(x, y - err, y + err, color=color, alpha=0.2, lw=0, zorder=0)
                    break


def timing_roi_figure(img, found):
    """FEL on-off image with the automatic ROI(s), and the x / y cuts through each
    peak with the ROI limits as vertical lines."""
    n = len(found)
    fig = Figure(figsize=(12, 3.8 * n), dpi=80)
    for k, a in enumerate(found):
        ax = fig.add_subplot(n, 3, 3 * k + 1)
        im = ax.imshow(img, cmap='jet', origin='upper', extent=(0.5, img.shape[1] + 0.5, img.shape[0] + 0.5, 0.5))
        fig.colorbar(im, ax=ax, fraction=0.05, label='counts / pixel')
        for j, b in enumerate(found):
            c, r, w, h = b['roi']
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), w, h, fill=False, ec=('#ffffff', '#f472b6')[j], lw=2))
        ax.plot(*a['peak'], '+', color='white', ms=12, mew=2)
        ax.set_title(f'FEL on - off, ROI {k + 1} = {fmt_list(a["roi"])}')
        for col, (cut, lim, lab) in enumerate(((a['cut_x'], a['x_limits'], 'x (column)'),
                                               (a['cut_y'], a['y_limits'], 'y (row)'))):
            ax = fig.add_subplot(n, 3, 3 * k + 2 + col)
            px = np.arange(1, cut.size + 1)
            ax.plot(px, cut, '-', color=('#e11d48', '#4f46e5')[col], lw=2)
            ax.axhline(a['level'], color='#f59e0b', ls='--', lw=1.2, label='background level')
            ax.axhline(0, color='#0f172a', lw=0.6)
            for v in lim:
                ax.axvline(v, color='#10b981', lw=2)
            ax.axvline(a['peak'][col], color='#a855f7', ls=':', lw=1.5, label='peak')
            ax.set_xlabel(lab)
            ax.set_title(f'cut along {lab.split()[0]} ({a["n_cuts"]} cuts averaged): {lim[0]} - {lim[1]}')
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

class Field:
    """Label + Entry bound to a StringVar, gridded in a parent."""

    def __init__(self, parent, row, label, value='', width=10, col=0, tip=None):
        self.label = label
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky='w', padx=2, pady=1)
        self.var = tk.StringVar(value=str(value))
        self.entry = ttk.Entry(parent, textvariable=self.var, width=width)
        self.entry.grid(row=row, column=col + 1, sticky='we', padx=2, pady=1)

    def get(self):
        return self.var.get().strip()

    def set(self, v):
        self.var.set(str(v))


def figure_window(master, title, figures, text=None):
    """Toplevel showing matplotlib Figures (with toolbars) and an optional text."""
    win = tk.Toplevel(master)
    win.title(title)
    win.configure(background=PALETTE['bg'])
    header(win, title)
    outer = ttk.Frame(win)
    outer.pack(fill='both', expand=True)
    for fig in figures:
        fr = ttk.Frame(outer)
        fr.pack(fill='both', expand=True)
        canvas = FigureCanvasTkAgg(fig, master=fr)
        canvas.draw()
        NavigationToolbar2Tk(canvas, fr).update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    if text:
        txt = console(outer, height=min(8, text.count('\n') + 2))
        txt.configure(wrap='none', font=('Courier', 10))
        txt.insert('1.0', text)
        txt.pack(fill='x')
    return win


def text_pages(title, text, lines=52):
    """A4 landscape pages of monospace text (settings, fit tables) for a PDF report."""
    rows = text.splitlines() or ['']
    pages = []
    for k in range(0, len(rows), lines):
        fig = Figure(figsize=(11.69, 8.27))
        n = len(rows) // lines + (len(rows) % lines > 0)
        fig.text(0.04, 0.96, title + (f'   ({k // lines + 1}/{n})' if n > 1 else ''), fontsize=13,
                 weight='bold', va='top', color=PALETTE['accent_dark'])
        chunk = rows[k:k + lines]
        size = 8.5 if max(map(len, chunk)) <= 150 else 6.5   # wide fit tables need the small font
        fig.text(0.04, 0.92, '\n'.join(chunk), family='monospace', fontsize=size, va='top')
        pages.append(fig)
    return pages


def write_pdf(path, pages):
    """One PDF page per Figure."""
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for fig in pages:
            pdf.savefig(fig)


# --------------------------------------------------------------------------
# main window: scan analysis
# --------------------------------------------------------------------------

def shot_stats_payload(results):
    """Per-file good-shot I0 and spectrometer integrals for the statistics tab."""
    out = []
    for r in results:
        g = np.asarray(r.good, bool)
        sp = r.spect_int if r.spect_int is not None else np.full(g.size, np.nan)
        out.append(dict(fnr=r.fnr, i0=np.asarray(r.iom_sh, float)[g], spect=np.asarray(sp, float)[g],
                        n_shots=r.n_shots, n_good=r.n_good, prbg=r.prbg))
    return out


class TimerGUI:
    def __init__(self, root):
        self.root = root
        root.title('TIMER analysis')
        apply_style(root)
        self.q = queue.Queue()
        self.worker = None
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.ana = None
        self.last_output = None
        self._build()
        self.load_settings(SETTINGS_FILE, quiet=True)
        self._update_option_states()
        root.after(150, self._poll)
        root.protocol('WM_DELETE_WINDOW', self.on_close)

    # ---- layout --------------------------------------------------------
    def _build(self):
        header(self.root, 'TIMER  ·  transient grating analysis', 'FERMI  EIS-TIMER   |   scan · statistics · background')
        left = ttk.Frame(self.root, padding=6)
        left.pack(side='left', fill='y')
        right = ttk.Frame(self.root, padding=6)
        right.pack(side='left', fill='both', expand=True)

        nb = ttk.Notebook(left)
        nb.pack(fill='x')
        scan = ttk.Frame(nb, padding=4)
        roi = ttk.Frame(nb, padding=4)
        nb.add(scan, text='Scan')
        nb.add(roi, text='ROI / spectrum')
        d = ta.AnalysisConfig()

        r = 0
        ttk.Label(scan, text='Experiment folder').grid(row=r, column=0, sticky='w')
        self.exp_folder = tk.StringVar(value=d.exp_folder)
        ttk.Entry(scan, textvariable=self.exp_folder, width=34).grid(row=r, column=1, sticky='we')
        ttk.Button(scan, text='…', width=3, command=lambda: self._browse_dir(self.exp_folder)).grid(row=r, column=2)
        r += 1
        self.f_scan_nr = Field(scan, r, 'Scan number', d.scan_nr); r += 1
        self.f_scan_str = Field(scan, r, 'Scan name format', d.scan_str); r += 1
        self.f_bg_str = Field(scan, r, 'OnlyProbe folder suffix', '_OnlyProbe'); r += 1
        self.f_bg_str2 = Field(scan, r, 'NoProbe folder suffix', '_NoProbe'); r += 1
        ttk.Label(scan, text='Output folder (blank = default)').grid(row=r, column=0, sticky='w')
        self.out_folder = tk.StringVar(value='')
        ttk.Entry(scan, textvariable=self.out_folder, width=34).grid(row=r, column=1, sticky='we')
        ttk.Button(scan, text='…', width=3, command=lambda: self._browse_dir(self.out_folder)).grid(row=r, column=2)
        r += 1
        ttk.Label(scan, text='x axis').grid(row=r, column=0, sticky='w')
        self.x_axis = tk.StringVar(value=d.x_axis)
        ttk.Combobox(scan, textvariable=self.x_axis, values=list(ta.X_AXES), width=34,
                     state='readonly').grid(row=r, column=1, columnspan=2, sticky='we')
        r += 1
        self.f_t0 = Field(scan, r, 'Time zero T0 (delay units)', d.t0); r += 1
        self.f_bin = Field(scan, r, 'Bin (ps)', d.bin); r += 1
        self.f_start_end = Field(scan, r, 'First, last file', fmt_list(d.start_end)); r += 1
        self.f_exclude = Field(scan, r, 'Exclude files', ''); r += 1
        self.f_exclude_bg = Field(scan, r, 'Exclude background files', ''); r += 1
        self.f_thr = Field(scan, r, 'FEL I0 threshold', d.fel_down_thr); r += 1
        ttk.Label(scan, text='Normalisation').grid(row=r, column=0, sticky='w')
        self.norm_mode = tk.StringVar(value=d.norm_mode)
        ttk.Combobox(scan, textvariable=self.norm_mode, values=ta.NORM_MODES, width=12,
                     state='readonly').grid(row=r, column=1, sticky='w')
        scan.columnconfigure(1, weight=1)

        r = 0
        self.f_roi = Field(roi, r, 'ROI col, row, width, height', fmt_list(d.roi), 18); r += 1
        self.f_ellipse = Field(roi, r, 'Ellipse axes (h, v)', fmt_list(d.ellipse), 18); r += 1
        self.f_bar_fix = Field(roi, r, 'Ellipse centre in ROI (x, y)', fmt_list(d.bar_fix), 18); r += 1
        ttk.Label(roi, text='Ellipse centre').grid(row=r, column=0, sticky='w')
        self.bar_mode = tk.StringVar(value=d.bar_mode)
        ttk.Combobox(roi, textvariable=self.bar_mode, values=ta.BAR_MODES, width=12,
                     state='readonly').grid(row=r, column=1, sticky='w')
        r += 1
        self.lock_bar = tk.BooleanVar(value=d.lock_bar)
        ttk.Checkbutton(roi, text='Lock centre after t > 0.3 ps', variable=self.lock_bar).grid(
            row=r, column=0, columnspan=2, sticky='w')
        r += 1
        self.bary_matlab = tk.BooleanVar(value=d.bary_matlab)
        ttk.Checkbutton(roi, text='Barycentre as old MATLAB (weights max − I; only to reproduce old files)',
                        variable=self.bary_matlab).grid(row=r, column=0, columnspan=2, sticky='w')
        r += 1
        self.f_prox = Field(roi, r, 'Proximity bg top, bottom, left, right', fmt_list(d.prox_cfg), 18); r += 1
        self.f_spect = Field(roi, r, 'Spectrum start, end, edge width', fmt_list(
            (d.spect_start, d.spect_end, d.d_spect)), 18); r += 1
        self.f_gain = Field(roi, r, 'CCD counts per photon (0 = no shot noise)', d.ccd_gain, 18); r += 1
        ttk.Button(roi, text='Estimate counts per photon from the last scan', style='Accent.TButton',
                   command=self.estimate_gain).grid(row=r, column=0, columnspan=2, sticky='we', padx=2, pady=2)
        r += 1

        # --- the three option boxes ------------------------------------
        opt = ttk.LabelFrame(left, text='Options', padding=6)
        opt.pack(fill='x', pady=4)
        self.use_bg = tk.BooleanVar(value=True)
        self.bg_choice = tk.StringVar(value='only')
        ttk.Checkbutton(opt, text='Subtract background', variable=self.use_bg, style='Head.TCheckbutton',
                        command=self._update_option_states).grid(row=0, column=0, columnspan=3, sticky='w')
        self.rb_bg = []
        for k, (val, text) in enumerate((
                ('only', 'OnlyProbe file, nearest bunch  (probe scatter)'),
                ('no', 'NoProbe file, nearest bunch  (FEL / pump scatter)'),
                ('both', 'NoProbe + OnlyProbe files, nearest bunch'),
                ('first_n', 'mean of the first N signal images, N ='))):
            rb = ttk.Radiobutton(opt, text=text, variable=self.bg_choice, value=val)
            rb.grid(row=1 + k, column=0, columnspan=2 if val == 'first_n' else 3, sticky='w', padx=18)
            self.rb_bg.append(rb)
        self.bg_n = tk.StringVar(value='2')
        self.e_bg_n = ttk.Entry(opt, textvariable=self.bg_n, width=5)
        self.e_bg_n.grid(row=4, column=2, sticky='w')

        self.single = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text='Single image only (set the ROI), file number', variable=self.single,
                        style='Head.TCheckbutton',
                        command=self._update_option_states).grid(row=5, column=0, columnspan=2, sticky='w')
        self.single_nr = tk.StringVar(value='10')
        self.e_single = ttk.Entry(opt, textvariable=self.single_nr, width=5)
        self.e_single.grid(row=5, column=2, sticky='w')

        self.loop = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text='Loop: add new files every (s)', variable=self.loop,
                        style='Head.TCheckbutton',
                        command=self._update_option_states).grid(row=6, column=0, columnspan=2, sticky='w')
        self.wait_s = tk.StringVar(value='30')
        self.e_wait = ttk.Entry(opt, textvariable=self.wait_s, width=5)
        self.e_wait.grid(row=6, column=2, sticky='w')
        self.fast = tk.BooleanVar(value=False)
        self.fast_flag = threading.Event()
        self.fast.trace_add('write', lambda *a: self.fast_flag.set() if self.fast.get() else self.fast_flag.clear())
        ttk.Checkbutton(opt, text='Fast run (unticked: pause on every file for (s))', variable=self.fast,
                        style='Head.TCheckbutton').grid(row=7, column=0, columnspan=2, sticky='w')
        # pause per file: arrows or typing, read live by the worker (change it during a run)
        self.show_s = tk.StringVar(value='0.3')
        self.pause_s = 0.3
        ttk.Spinbox(opt, textvariable=self.show_s, from_=0, to=10, increment=0.1, width=5,
                    format='%.1f').grid(row=7, column=2, sticky='w')
        self.show_s.trace_add('write', self._update_pause)

        btn = ttk.Frame(left)
        btn.pack(fill='x')
        self.b_run = ttk.Button(btn, text='▶  Run', command=self.run, style='Run.TButton')
        self.b_run.grid(row=0, column=0, sticky='we', padx=2, pady=2)
        self.b_stop = ttk.Button(btn, text='■  Stop (save)', command=self.stop, state='disabled',
                                 style='Stop.TButton')
        self.b_stop.grid(row=0, column=1, sticky='we', padx=2, pady=2)
        self.b_abort = ttk.Button(btn, text='✕  Abort (no save)', command=self.abort, state='disabled',
                                  style='Abort.TButton')
        self.b_abort.grid(row=0, column=2, sticky='we', padx=2, pady=2)
        ttk.Button(btn, text='Save settings…', command=self.save_settings_as).grid(row=1, column=0, sticky='we',
                                                                                  padx=2, pady=2)
        ttk.Button(btn, text='Load settings…', command=self.load_settings_from).grid(row=1, column=1, sticky='we',
                                                                                  padx=2, pady=2)
        ttk.Button(btn, text='📄  PDF report', command=self.scan_report).grid(row=1, column=2, sticky='we',
                                                                             padx=2, pady=2)
        ttk.Button(btn, text='⏱  Timing (Svirgo)…', command=self.open_timing, style='Tool.TButton').grid(
            row=2, column=0, sticky='we', padx=2, pady=2)
        ttk.Button(btn, text='∿  Trace analysis…', command=self.open_trace, style='Accent.TButton').grid(
            row=2, column=1, columnspan=2, sticky='we', padx=2, pady=2)
        for k in range(3):
            btn.columnconfigure(k, weight=1)

        self.status = tk.StringVar(value='Ready')
        self.status_label = tk.Label(left, textvariable=self.status, wraplength=380, justify='left', anchor='w',
                                     background=STATUS_COLORS['idle'], foreground=PALETTE['accent_dark'],
                                     font=('Helvetica', 10, 'bold'), padx=8, pady=6)
        self.status_label.pack(fill='x', pady=6)
        self.status.trace_add('write', lambda *a: self._color_status())
        self.log_box = console(left)
        self.log_box.pack(fill='both', expand=True)

        tabs = ttk.Notebook(right)
        tabs.pack(fill='both', expand=True)
        self.tabs = tabs

        def tab(title):
            fr = ttk.Frame(tabs)
            tabs.add(fr, text=title)
            fig = Figure(figsize=(13, 9), dpi=80)
            canvas = FigureCanvasTkAgg(fig, master=fr)
            NavigationToolbar2Tk(canvas, fr).update()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            return fig, canvas

        self.fig, self.canvas = tab('Scan')
        self.fig_stats, self.canvas_stats = tab('I0 / spectrometer statistics')
        self.fig_bg, self.canvas_bg = tab('Background')
        # only the visible tab is redrawn per file (keeps Stop/Abort responsive);
        # the others are drawn from the latest data when they are opened
        self._pending = {}
        self.shown = threading.Event()
        tabs.bind('<<NotebookTabChanged>>', lambda e: self._draw_pending())

    def _browse_dir(self, var):
        d = filedialog.askdirectory(initialdir=var.get() or os.getcwd())
        if d:
            var.set(d)

    def _color_status(self):
        t = self.status.get().lower()
        key = ('bad' if t.startswith(('abort', 'error')) else 'ok' if t.startswith(('written', 'single'))
               else 'warn' if t.startswith(('stop', 'waiting')) else 'idle' if t.startswith('ready') else 'busy')
        self.status_label.configure(background=STATUS_COLORS[key])

    def estimate_gain(self):
        """Fit the counts per photon to the scatter of the scan analysed last."""
        if self.ana is None or len(self.ana.results) < 15:
            messagebox.showinfo('CCD counts per photon', 'Analyse a scan first (at least ~15 points after t0).')
            return
        try:
            g, lo, hi, n = ta.estimate_ccd_gain(list(self.ana.results.values()))
        except ValueError as e:
            messagebox.showinfo('CCD counts per photon', str(e))
            return
        self.f_gain.set(f'{g:.1f}')
        self.log(f'CCD counts per photon from the scatter of {n} points: {g:.1f} (1 sigma {lo:.1f} - {hi:.1f}). '
                 'Run again to apply it to the error bands.')

    def _update_option_states(self):
        st = 'normal' if self.use_bg.get() else 'disabled'
        for w in (*self.rb_bg, self.e_bg_n):
            w.configure(state=st)
        self.e_single.configure(state='normal' if self.single.get() else 'disabled')
        self.e_wait.configure(state='normal' if self.loop.get() else 'disabled')

    def log(self, msg):
        self.q.put(('log', msg))

    # ---- settings <-> config ------------------------------------------
    def config(self) -> ta.AnalysisConfig:
        spect = parse_list(self.f_spect.get(), int)
        return ta.AnalysisConfig(
            exp_folder=self.exp_folder.get().strip(),
            scan_nr=int(self.f_scan_nr.get()),
            scan_str=self.f_scan_str.get(),
            bg_str=self.f_bg_str2.get() if self.bg_choice.get() == 'no' else self.f_bg_str.get(),
            bg_str2=self.f_bg_str2.get(),
            out_folder=self.out_folder.get().strip(),
            bg_mode=({'only': 'bunch', 'no': 'bunch', 'both': 'both', 'first_n': 'first_n'}[self.bg_choice.get()]
                     if self.use_bg.get() else 'none'),
            bg_first_n=int(self.bg_n.get()),
            single_image=self.single.get(),
            single_image_nr=int(self.single_nr.get()),
            loop=self.loop.get(),
            wait_s=float(self.wait_s.get()),
            show_s=float(self.show_s.get() or 0),
            fast_run=self.fast.get(),
            start_end=tuple(parse_list(self.f_start_end.get(), int)[:2]),
            exclude=tuple(parse_list(self.f_exclude.get(), int)),
            exclude_bg=tuple(parse_list(self.f_exclude_bg.get(), int)),
            fel_down_thr=float(self.f_thr.get()),
            t0=float(self.f_t0.get()),
            bin=float(self.f_bin.get()),
            x_axis=self.x_axis.get(),
            roi=tuple(parse_list(self.f_roi.get(), int)),
            ellipse=tuple(parse_list(self.f_ellipse.get(), float)),
            bar_fix=tuple(parse_list(self.f_bar_fix.get(), int)),
            bar_mode=self.bar_mode.get(),
            lock_bar=self.lock_bar.get(),
            bary_matlab=self.bary_matlab.get(),
            prox_cfg=tuple(parse_list(self.f_prox.get(), int)),
            spect_start=spect[0], spect_end=spect[1], d_spect=spect[2],
            norm_mode=self.norm_mode.get(),
            ccd_gain=float(self.f_gain.get() or 0),
        )

    def apply_config(self, c: ta.AnalysisConfig):
        self.exp_folder.set(c.exp_folder)
        self.f_scan_nr.set(c.scan_nr)
        self.f_scan_str.set(c.scan_str)
        self.f_bg_str2.set(c.bg_str2)
        no_probe = c.bg_mode == 'bunch' and 'noprobe' in c.bg_str.lower()
        if no_probe:
            self.f_bg_str2.set(c.bg_str)
        else:
            self.f_bg_str.set(c.bg_str)
        self.out_folder.set(c.out_folder)
        self.use_bg.set(c.bg_mode != 'none')
        self.bg_choice.set({'bunch': 'no' if no_probe else 'only', 'both': 'both', 'first_n': 'first_n'}.get(
            c.bg_mode, 'only'))
        self.bg_n.set(c.bg_first_n)
        self.single.set(c.single_image)
        self.single_nr.set(c.single_image_nr)
        self.loop.set(c.loop)
        self.wait_s.set(c.wait_s)
        self.show_s.set(c.show_s)
        self.fast.set(c.fast_run)
        self.f_start_end.set(fmt_list(c.start_end))
        self.f_exclude.set(fmt_list(c.exclude))
        self.f_exclude_bg.set(fmt_list(c.exclude_bg))
        self.f_thr.set(c.fel_down_thr)
        self.f_t0.set(c.t0)
        self.f_bin.set(c.bin)
        if c.x_axis in ta.X_AXES:
            self.x_axis.set(c.x_axis)
        self.f_roi.set(fmt_list(c.roi))
        self.f_ellipse.set(fmt_list(c.ellipse))
        self.f_bar_fix.set(fmt_list(c.bar_fix))
        self.bar_mode.set(c.bar_mode)
        self.lock_bar.set(c.lock_bar)
        self.bary_matlab.set(c.bary_matlab)
        self.f_prox.set(fmt_list(c.prox_cfg))
        self.f_spect.set(fmt_list((c.spect_start, c.spect_end, c.d_spect)))
        self.norm_mode.set(c.norm_mode)
        self.f_gain.set(c.ccd_gain)
        self._update_option_states()

    def save_settings_as(self):
        p = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if p:
            self.config().save(p)

    def load_settings_from(self):
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json'), ('All', '*')])
        if p:
            self.load_settings(p)

    def load_settings(self, path, quiet=False):
        try:
            self.apply_config(ta.AnalysisConfig.load(path))
        except Exception as e:
            if not quiet:
                messagebox.showerror('Load settings', str(e))

    # ---- run / stop ----------------------------------------------------
    def run(self):
        try:
            cfg = self.config()
        except (ValueError, IndexError) as e:
            messagebox.showerror('Settings', f'Check the settings: {e}')
            return
        if len(cfg.roi) != 4 or len(cfg.ellipse) != 2 or len(cfg.bar_fix) != 2 or len(cfg.prox_cfg) != 4:
            messagebox.showerror('Settings', 'ROI needs 4 numbers, ellipse and centre 2, proximity 4.')
            return
        try:
            cfg.save(SETTINGS_FILE)
        except OSError:
            pass
        self.stop_event.clear()
        self.abort_event.clear()
        self.b_run.configure(state='disabled')
        self.b_stop.configure(state='normal')
        self.b_abort.configure(state='normal')
        self.status.set('Starting…')
        self.worker = threading.Thread(target=self._work, args=(cfg,), daemon=True)
        self.worker.start()

    def stop(self):
        """Finish the current file, then bin and save everything analysed so far."""
        self.stop_event.set()
        self.status.set('Stopping after the current file, then saving…')

    def abort(self):
        """Stop after the current file and write nothing."""
        self.abort_event.set()
        self.stop_event.set()
        self.status.set('Aborting, nothing will be saved…')

    def _work(self, cfg):
        try:
            ana = ta.ScanAnalysis(cfg, log=self.log)
            self.ana = ana
            self.log(f'{ana.dataset}: {ana.n_files} files, {ana.n_files_bg} background files, '
                     f'{ana.fel}, {ana.bl}')
            if cfg.single_image:
                fnr = min(cfg.single_image_nr, ana.n_files)
                while True:   # a file without a CCD image: show the next one
                    try:
                        res = ana.process_file(fnr, roi_only=True)
                        break
                    except ta.SkipFile as e:
                        if 'no CCD image' not in str(e) or fnr >= ana.n_files:
                            raise
                        self.log(f'{e}: showing file {fnr + 1}')
                        fnr += 1
                self.q.put(('single', res, ana, shot_stats_payload([res])))
                return
            while True:
                def on_file(res):
                    rs = [ana.results[k] for k in sorted(ana.results)]
                    mnf = np.mean([r.mean_norm_factor for r in rs])
                    summary = [(r.fnr, r.time, r.tg_sum_nor, r.cg, r.rg, r.cbar, r.rbar,
                                r.tg_sum_nor * mnf, r.tg_nor_err * mnf) for r in rs]
                    self.shown.clear()
                    t_put = time.time()
                    self.q.put(('file', res, summary, cfg, shot_stats_payload(rs)))
                    # wait until the GUI has drawn this file, and keep it on screen for the pause
                    if not self.fast_flag.is_set():   # read live: can be ticked during a run
                        self.shown.wait(timeout=5)
                        while (time.time() - t_put < self.pause_s and not self.stop_event.is_set()
                               and not self.fast_flag.is_set()):
                            time.sleep(0.02)
                ana.refresh_lists()
                todo = ana.files_to_process()
                self.q.put(('status', f'Analysing {len(todo)} file(s) of {ana.n_files}…'))
                ana.process_files(todo, on_file=on_file, stop=self.stop_event.is_set)
                if self.abort_event.is_set():
                    self.q.put(('log', 'aborted: nothing written for this pass'))
                    self.q.put(('status', 'Aborted (nothing saved)'))
                    break
                if ana.results:
                    R = ana.bin_results()
                    bg = ana.check_background()
                    fp = ana.write_outputs()
                    self.q.put(('pass', R, bg, fp, cfg))
                if not cfg.loop or self.stop_event.is_set():
                    break
                for k in range(int(cfg.wait_s)):
                    if self.stop_event.is_set():
                        break
                    self.q.put(('status', f'Waiting for new files {k + 1}/{int(cfg.wait_s)} s'))
                    time.sleep(1)
                if self.stop_event.is_set():
                    break
        except ta.SkipFile as e:
            self.q.put(('error', str(e), str(e)))
        except Exception as e:
            self.q.put(('error', f'{type(e).__name__}: {e}', traceback.format_exc()))
        finally:
            self.q.put(('done',))

    # ---- queue -> plots ------------------------------------------------
    def _poll(self):
        latest_file = None
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == 'log':
                    self.log_box.insert('end', msg[1] + '\n')
                    self.log_box.see('end')
                elif kind == 'status':
                    self.status.set(msg[1])
                elif kind == 'file':
                    latest_file = msg
                elif kind == 'single':
                    self._queue_draw(1, lambda m=msg: self.plot_stats(m[1], m[3]))
                    self._queue_draw(2, lambda m=msg: self.plot_background(m[1], m[3], m[2].cfg))
                    self._queue_draw(0, lambda m=msg: self.plot_single(m[1], m[2]))
                    self.status.set(f'Single image: file {msg[1].fnr}')
                elif kind == 'pass':
                    if latest_file:
                        self.plot_file(*latest_file[1:])
                        latest_file = None
                    R, bg, c = msg[1], msg[2], msg[4]
                    scan_job = self._pending.get(0)
                    self._queue_draw(0, (lambda j=scan_job, R=R, bg=bg, c=c: (j and j(), self.plot_binned(R, bg, c))))
                    bg_job = self._pending.get(2)
                    self._queue_draw(2, (lambda j=bg_job, bg=bg, c=c: (j and j(), self.plot_bg_check(bg, c))))
                    self.last_output = msg[3]
                    self.status.set(f'Written {msg[3]}_*.txt')
                    self.log_box.insert('end', f'written {msg[3]}_nor.txt\n')
                    self.log_box.see('end')
                elif kind == 'error':
                    self.log_box.insert('end', msg[2] + '\n')
                    self.log_box.see('end')
                    messagebox.showerror('Analysis', msg[1])
                elif kind == 'done':
                    self.b_run.configure(state='normal')
                    self.b_stop.configure(state='disabled')
                    self.b_abort.configure(state='disabled')
                    if self.abort_event.is_set():
                        self.status.set('Aborted (nothing saved)')
                    elif not self.status.get().startswith(('Written', 'Single')):
                        self.status.set('Stopped')
        except queue.Empty:
            pass
        if latest_file:
            self.plot_file(*latest_file[1:])
            if not self.fast.get():
                self.canvas.draw()
            self.root.after_idle(self.shown.set)
        self.root.after(50, self._poll)

    def scan_report(self):
        """Scan_NNN_report.pdf next to the output files: settings and summary,
        then the Scan, statistics and background tabs as they are now."""
        if not self.last_output or self.ana is None:
            messagebox.showinfo('PDF report', 'Run a scan first: the report goes next to its output files.')
            return
        for k in (1, 2):   # draw the tabs that were not looked at
            job = self._pending.pop(k, None)
            if job:
                job()
        ana, cfg = self.ana, self.ana.cfg
        rs = [ana.results[k] for k in sorted(ana.results)]
        from dataclasses import asdict
        lines = [f'{ana.experiment}  {ana.dataset}', time.strftime('report written %Y-%m-%d %H:%M'),
                 f'output: {self.last_output}_*.txt', f'files analysed: {len(rs)} of {ana.n_files}, '
                 f'background files: {ana.n_files_bg}, {ana.fel}, {ana.bl}',
                 f'good shots: {sum(r.n_good for r in rs)} / {sum(r.n_shots for r in rs)}', '',
                 'settings (also in ' + os.path.basename(self.last_output) + '_parameters.json):',
                 *(f'  {k:<16} {v}' for k, v in asdict(cfg).items())]
        path = self.last_output + '_report.pdf'
        try:
            write_pdf(path, text_pages(f'TIMER scan report: {ana.dataset}', '\n'.join(lines))
                      + [self.fig, self.fig_stats, self.fig_bg])
        except Exception as e:
            messagebox.showerror('PDF report', f'{type(e).__name__}: {e}')
            return
        for c in (self.canvas, self.canvas_stats, self.canvas_bg):
            c.draw_idle()
        self.log(f'report written: {path}')
        self.status.set(f'Report written: {path}')

    def _update_pause(self, *args):
        """Pause per file, read live by the worker thread (a float is safe to share)."""
        try:
            self.pause_s = max(0.0, float(self.show_s.get()))
        except ValueError:
            pass   # half-typed number: keep the last valid pause

    def _axes(self, n_rows=3, n_cols=3):
        self.fig.clear()
        return [self.fig.add_subplot(n_rows, n_cols, k + 1) for k in range(n_rows * n_cols)]

    def _draw_image_with_rois(self, ax, image, cfg, log=True):
        show_stretched(ax, image)
        c, r, w, h = cfg.roi
        ax.add_patch(Rectangle((c, r), w, h, fill=False, ec='w', lw=2))
        for p, col in zip(ta.prox_roi_param(cfg.roi, cfg.prox_cfg), ('c', 'g', 'm', 'y')):
            ax.add_patch(Rectangle((p[0], p[1]), p[2], p[3], fill=False, ec=col, lw=1))
        ax.set_title('Full image (linear, contrast enhanced)')

    def _draw_zoom(self, ax, res, cfg, title='Zoom image'):
        show_stretched(ax, res.zoom)
        for col, lw in (('r', 3), ('w', 1)):
            ax.add_patch(Ellipse((res.cbar, res.rbar), cfg.ellipse[0], cfg.ellipse[1], fill=False, ec=col, lw=lw))
        ax.set_title(title)

    def _draw_shots(self, ax, res):
        xs = np.arange(1, res.n_shots + 1)
        ax.plot(xs, res.iom_sh, 'o', ms=3, color='#e11d48', label='rejected')
        ax.plot(xs[res.good], res.iom_sh[res.good], '.', color='#10b981', label='good')
        ax.set_xlim(1, res.n_shots)
        ax.set_title('IoM Sh')

    def _draw_spectrum(self, ax, res, cfg, pixels=False):
        if pixels or np.ndim(res.x_spect) != 1 or len(res.x_spect) < cfg.spect_end:
            ax.plot(np.arange(cfg.spect_start, cfg.spect_end + 1), res.spect_ave)
            ax.set_xlabel('Pixel index')
        else:
            xs = res.x_spect[cfg.spect_start - 1:cfg.spect_end]
            ax.plot(xs, res.spect_ave)
            ax.set_xlabel('Wavelength (nm)')
            ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title('Spectrum')

    def plot_single(self, res, ana):
        cfg = ana.cfg
        self.fig.clear()
        axs = [self.fig.add_subplot(2, 2, k + 1) for k in range(4)]
        self._draw_shots(axs[0], res)
        self._draw_spectrum(axs[1], res, cfg, pixels=True)
        self._draw_image_with_rois(axs[2], res.image_sub, cfg)
        self._draw_zoom(axs[3], res, cfg)
        self._ell_edit(axs[3], res, cfg)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---- single image: move / resize the ellipse with the mouse ----------
    def _ell_edit(self, ax, res, cfg):
        """Drag the ellipse in the zoom image to move it; scroll changes the
        horizontal axis, shift + scroll the vertical one. On release the
        'Ellipse centre' / 'Ellipse axes' fields are updated (and the centre
        mode set to fixed), so the next run and the saved parameters use them."""
        for cid in getattr(self, '_ell_cids', []):
            self.canvas.mpl_disconnect(cid)
        pats = [p for p in ax.patches if isinstance(p, Ellipse)]
        st = dict(ax=ax, pats=pats, centre=[float(res.cbar), float(res.rbar)],
                  axes=[float(cfg.ellipse[0]), float(cfg.ellipse[1])], drag=None, fnr=res.fnr)
        self._ell_state = st
        self._ell_title()
        self._ell_cids = [self.canvas.mpl_connect('button_press_event', self._ell_press),
                          self.canvas.mpl_connect('motion_notify_event', self._ell_motion),
                          self.canvas.mpl_connect('button_release_event', self._ell_release),
                          self.canvas.mpl_connect('scroll_event', self._ell_scroll)]

    def _ell_live(self, event):
        st = getattr(self, '_ell_state', None)
        if st is None or st['ax'] not in self.fig.axes:   # another plot replaced the single image
            return None
        return st

    def _ell_title(self):
        st = self._ell_state
        (cx, cy), (h, v) = st['centre'], st['axes']
        st['ax'].set_title(f'Zoom, file {st["fnr"]}: centre ({cx:.0f}, {cy:.0f}), axes ({h:.0f}, {v:.0f})\n'
                           'drag = move, scroll = width, shift+scroll = height', fontsize=9)

    def _ell_redraw(self):
        st = self._ell_state
        for p in st['pats']:
            p.set_center(st['centre']); p.width, p.height = st['axes']
        self._ell_title()
        self.canvas.draw_idle()

    def _ell_press(self, event):
        st = self._ell_live(event)
        if st is None or event.inaxes is not st['ax'] or event.button != 1 or event.xdata is None:
            return
        tb = getattr(self.canvas, 'toolbar', None)
        if tb is not None and str(tb.mode):   # zoom / pan tool active: leave the mouse to it
            return
        st['drag'] = (event.xdata - st['centre'][0], event.ydata - st['centre'][1])

    def _ell_motion(self, event):
        st = self._ell_live(event)
        if st is None or st['drag'] is None or event.inaxes is not st['ax'] or event.xdata is None:
            return
        st['centre'] = [event.xdata - st['drag'][0], event.ydata - st['drag'][1]]
        self._ell_redraw()

    def _ell_release(self, event):
        st = self._ell_live(event)
        if st is None or st['drag'] is None:
            return
        st['drag'] = None
        st['centre'] = [float(round(c)) for c in st['centre']]
        self._ell_redraw()
        self._ell_store()

    def _ell_scroll(self, event):
        st = self._ell_live(event)
        if st is None or event.inaxes is not st['ax']:
            return
        k = 1 if (event.key or '').lower().startswith('shift') else 0
        step = 1 if event.button == 'up' else -1
        st['axes'][k] = max(2.0, st['axes'][k] + step)
        self._ell_redraw()
        self._ell_store()

    def _ell_store(self):
        st = self._ell_state
        cx, cy = (int(round(c)) for c in st['centre'])
        h, v = (int(round(a)) for a in st['axes'])
        self.f_bar_fix.set(fmt_list((cx, cy)))
        self.f_ellipse.set(fmt_list((h, v)))
        if self.bar_mode.get() != 'fixed':
            self.bar_mode.set('fixed')
            self.log('ellipse centre mode set to fixed (centre moved by hand)')
        self.log(f'ellipse: centre ({cx}, {cy}), axes ({h}, {v}) written to the ROI fields')

    def _tab_index(self):
        try:
            return self.tabs.index(self.tabs.select())
        except tk.TclError:
            return 0

    def _draw_pending(self):
        k = self._tab_index()
        job = self._pending.pop(k, None)
        if job:
            job()

    def _queue_draw(self, tab_index, job):
        self._pending[tab_index] = job
        if self._tab_index() == tab_index:
            self._draw_pending()

    def plot_file(self, res, summary, cfg, stats=None):
        if stats is not None:
            self._queue_draw(1, lambda: self.plot_stats(res, stats))
            self._queue_draw(2, lambda: self.plot_background(res, stats, cfg))
        self._queue_draw(0, lambda: self._plot_scan(res, summary, cfg))

    def _plot_scan(self, res, summary, cfg):
        """Scan tab for one file. The 9 panels are built once per run and then
        only their data are swapped, so a redraw stays well under a second and
        the pause per file really sets the pace."""
        art = getattr(self, '_scan_art', None)
        key = (cfg.roi, cfg.prox_cfg, cfg.ellipse, cfg.x_axis)
        if art is None or art['key'] != key or art['axs'][0] not in self.fig.axes:
            art = self._build_scan(res, summary, cfg, key)
        axs = art['axs']
        xs = np.arange(1, res.n_shots + 1)
        art['shots_all'].set_data(xs, res.iom_sh)
        art['shots_good'].set_data(xs[res.good], res.iom_sh[res.good])
        axs[0].set_xlim(1, max(res.n_shots, 2))
        sx, sy = self._spectrum_xy(res, cfg)
        art['spect'].set_data(sx, sy)
        for k, img in ((2, res.image_sub), (3, res.zoom)):
            disp, lo, hi = stretch(img)
            art['im'][k].set_data(disp)
            art['im'][k].set_clim(lo, hi)
        art['ell'][0].set_center((res.cbar, res.rbar))
        art['ell'][1].set_center((res.cbar, res.rbar))
        axs[3].set_title(f'Zoom image, file {res.fnr}')
        s = np.array(summary, float)
        fn, tt_ = s[:, 0], s[:, 1]
        top_ok = bool(len(fn) > 1 and np.all(np.diff(fn) > 0)
                      and (np.all(np.diff(tt_) > 0) or np.all(np.diff(tt_) < 0)))
        if top_ok:   # time axis on top of "TG signal vs file", only when file -> time is monotonic
            art['fn'], art['tt'] = fn, tt_
        art['top'].set_visible(top_ok)
        set_error_band(art['band4'], s[:, 0], s[:, 7], s[:, 8])
        set_error_band(art['band5'], s[:, 1], s[:, 7], s[:, 8])
        ex, ey = cfg.ellipse
        art['sx'].set_data(s[:, 1], s[:, 3] - ex / 2 - 1)
        art['sy'].set_data(s[:, 1], s[:, 4] - ey / 2 - 1)
        for ax in (axs[0], axs[1], axs[4], axs[5], axs[6]):
            ax.relim()
        for h in (art['band4'], art['band5']):
            ok = np.all(np.isfinite(h['lim']), axis=1)
            if ok.any():
                h['ax'].update_datalim(h['lim'][ok])
        for ax in (axs[0], axs[1], axs[4], axs[5], axs[6]):
            ax.autoscale_view()
        self.canvas.draw_idle()
        if not self.abort_event.is_set() and str(self.b_run['state']) == 'disabled':
            self.status.set(f'File {res.fnr}: {res.n_good}/{res.n_shots} good shots')

    def _spectrum_xy(self, res, cfg):
        if np.ndim(res.x_spect) != 1 or len(res.x_spect) < cfg.spect_end:
            return np.arange(cfg.spect_start, cfg.spect_end + 1), res.spect_ave
        return res.x_spect[cfg.spect_start - 1:cfg.spect_end], res.spect_ave

    def _build_scan(self, res, summary, cfg, key):
        axs = self._axes()
        art = {'key': key, 'axs': axs, 'im': {}}
        art['shots_all'], = axs[0].plot([], [], 'o', ms=3, color='#e11d48', label='rejected')
        art['shots_good'], = axs[0].plot([], [], '.', color='#10b981', label='good')
        axs[0].set_title('IoM Sh')
        art['spect'], = axs[1].plot([], [])
        pixels = np.ndim(res.x_spect) != 1 or len(res.x_spect) < cfg.spect_end
        axs[1].set_xlabel('Pixel index' if pixels else 'Wavelength (nm)')
        if not pixels:
            axs[1].xaxis.set_major_locator(MaxNLocator(4))
        axs[1].set_title('Spectrum')
        art['im'][2] = show_stretched(axs[2], res.image_sub)
        c, r, w, h = cfg.roi
        axs[2].add_patch(Rectangle((c, r), w, h, fill=False, ec='w', lw=2))
        for p, col in zip(ta.prox_roi_param(cfg.roi, cfg.prox_cfg), ('c', 'g', 'm', 'y')):
            axs[2].add_patch(Rectangle((p[0], p[1]), p[2], p[3], fill=False, ec=col, lw=1))
        axs[2].set_title('Full image (linear, contrast enhanced)')
        art['im'][3] = show_stretched(axs[3], res.zoom)
        art['ell'] = [axs[3].add_patch(Ellipse((res.cbar, res.rbar), cfg.ellipse[0], cfg.ellipse[1],
                                               fill=False, ec=col, lw=lw)) for col, lw in (('r', 3), ('w', 1))]
        xlab = 'M2Z (mm)' if cfg.x_axis.startswith('M2Z') else ('Phase' if 'Phase' in cfg.x_axis else 'Time (ps)')
        art['band4'] = error_band(axs[4], [], [], [], '#f97316', marker='s')
        axs[4].set_title('TG signal vs file (band: CCD + normalisation error)')
        axs[4].set_xlabel('File number')
        axs[4].set_ylabel('Counts per shot')
        top = axs[4].secondary_xaxis('top', functions=(lambda f: np.interp(f, art['fn'], art['tt']),
                                                       lambda t: np.interp(t, np.sort(art['tt']),
                                                                           art['fn'][np.argsort(art['tt'])])))
        top.set_xlabel(xlab, fontsize=8)
        top.tick_params(labelsize=7)
        top.xaxis.set_major_locator(MaxNLocator(6))
        art['top'] = top
        art['fn'] = art['tt'] = np.array([0.0, 1.0])
        art['band5'] = error_band(axs[5], [], [], [], '#e11d48')
        axs[5].set_title('TG signal (band: CCD + normalisation error)')
        axs[5].set_xlabel(xlab)
        axs[5].grid(True)
        art['sx'], = axs[6].plot([], [], '-', color='#4f46e5', label='X')
        art['sy'], = axs[6].plot([], [], '-', color='#f97316', label='Y')
        axs[6].set_title('Signal shift (barycentre - ellipse centre)')
        axs[6].legend(fontsize=8)
        axs[6].set_xlabel(xlab)
        for ax in axs[7:]:
            ax.set_visible(False)
        self.fig.tight_layout()
        self._scan_art = art
        return art

    def plot_binned(self, R, bg, cfg):
        art = getattr(self, '_scan_art', None)
        axs = art['axs'] if art else []
        if len(axs) < 9 or axs[0] not in self.fig.axes or R is None:
            return
        xlab = axs[5].get_xlabel()
        ax = axs[7]
        ax.set_visible(True)
        ax.clear()
        error_band(ax, R['Time'][:, 0], R['TGSignal'][:, 0], R['TGSignalErr'], '#e11d48', ms=5,
                   label='binned TG, band = propagated error')
        ax.legend(fontsize=7, loc='best')
        ax.set_xlabel(xlab)
        ax.set_ylabel('Counts per shot')
        ax.set_title(f'{self.ana.experiment}  {self.ana.dataset}'.replace('_', ' '))
        ax.grid(True)
        ax = axs[8]
        ax.clear()
        if bg is not None and len(bg[1]):
            ax.set_visible(True)
            ax.plot(bg[1], '-o', color='#0ea5e9')
            ax.set_title('Background variation')
        else:
            ax.set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---- statistics and background tabs ----------------------------------
    def plot_stats(self, res, stats):
        """Shot-to-shot I0 and spectrometer integral, accumulated over all the
        files analysed so far (good shots only), plus the current file."""
        fig = self.fig_stats
        fig.clear()
        axs = [fig.add_subplot(2, 3, k + 1) for k in range(6)]
        if not stats:
            self.canvas_stats.draw_idle()
            return
        fnr = np.array([s['fnr'] for s in stats])
        i0_all = np.concatenate([s['i0'] for s in stats])
        sp_all = np.concatenate([s['spect'] for s in stats])
        cur = next((s for s in stats if s['fnr'] == res.fnr), stats[-1])
        colours = {'i0': dict(fill='tab:blue', band='lightskyblue', line='navy'),
                   'spect': dict(fill='darkorange', band='navajowhite', line='saddlebrown')}
        for row, key, name in ((0, 'i0', 'I0 (iom_sh_a)'), (1, 'spect', 'Spectrometer integral')):
            c = colours[key]
            allv = i0_all if key == 'i0' else sp_all
            ax = axs[3 * row]
            bins = np.histogram_bin_edges(allv, bins=min(60, max(10, int(np.sqrt(allv.size)))))
            ax.hist(allv, bins=bins, color=c['fill'], alpha=0.75, label=f'all files ({allv.size} shots)')
            ax.hist(cur[key], bins=bins, histtype='step', color='crimson', lw=1.8, label=f'file {cur["fnr"]}')
            m, sd = allv.mean(), allv.std(ddof=1) if allv.size > 1 else 0.0
            ax.axvline(m, color=c['line'], lw=1.5)
            ax.set_title(f'{name}: mean {m:.4g}, std {sd:.3g} ({100 * sd / m if m else np.nan:.1f}%)', fontsize=9)
            ax.legend(fontsize=7)
            ax = axs[3 * row + 1]
            mean = np.array([s[key].mean() if s[key].size else np.nan for s in stats])
            std = np.array([s[key].std(ddof=1) if s[key].size > 1 else 0.0 for s in stats])
            ax.axhspan(m - sd, m + sd, color=c['band'], alpha=0.6)
            ax.axhline(m, color=c['line'], lw=1.5)
            ax.errorbar(fnr, mean, std, fmt='o', ms=3, color=c['fill'], ecolor=c['fill'], elinewidth=0.8)
            ax.plot(cur['fnr'], cur[key].mean(), 'o', ms=6, color='crimson')
            ax.set_xlabel('File number')
            ax.set_title(f'{name} per file: mean ± std (band: all shots ±1σ)', fontsize=9)
        ax = axs[2]
        ok = np.isfinite(i0_all) & np.isfinite(sp_all)
        xi, yi = i0_all[ok], sp_all[ok]
        if xi.size > 2:
            h, xe, ye = np.histogram2d(xi, yi, bins=80)
            ix = np.clip(np.searchsorted(xe, xi) - 1, 0, h.shape[0] - 1)
            iy = np.clip(np.searchsorted(ye, yi) - 1, 0, h.shape[1] - 1)
            dens = h[ix, iy]
            order = np.argsort(dens)
            sc = ax.scatter(xi[order], yi[order], c=dens[order], cmap='jet', s=3, lw=0)
            fig.colorbar(sc, ax=ax, fraction=0.046, label='shots per bin')
        ax.plot(cur['i0'], cur['spect'], 'o', ms=2.5, mfc='none', mec='k', mew=0.5, alpha=0.6, label=f'file {cur["fnr"]}')
        ax.legend(fontsize=7, loc='upper left')
        r = np.corrcoef(xi, yi)[0, 1] if xi.size > 2 else np.nan
        ax.set_xlabel('I0')
        ax.set_ylabel('Spectrometer integral')
        ax.set_title(f'I0 vs spectrometer integral (density), r = {r:.3f}', fontsize=9)
        ax = axs[5]
        good = np.array([s['n_good'] / s['n_shots'] for s in stats]) * 100
        ax.plot(fnr, good, '-o', ms=3, color='#10b981')
        ax.set_ylim(0, 105)
        ax.set_xlabel('File number')
        ax.set_title(f'Good shots per file (%); total {sum(s["n_good"] for s in stats)}'
                     f'/{sum(s["n_shots"] for s in stats)}', fontsize=9)
        fig.tight_layout()
        self.canvas_stats.draw_idle()

    def plot_background(self, res, stats, cfg):
        """Spectrometer window used for the normalisation, background image
        subtracted from this file, proximity background per file."""
        fig = self.fig_bg
        fig.clear()
        axs = [fig.add_subplot(2, 2, k + 1) for k in range(4)]
        ax = axs[0]
        if res.spect_full is not None:
            px = np.arange(1, res.spect_full.size + 1)
            ax.plot(px, res.spect_full, 'k', lw=1, label='raw mean spectrum (good shots)')
            ax.axvspan(cfg.spect_start, cfg.spect_end, color='y', alpha=0.3, label='normalisation window')
            for a, b in ((cfg.spect_start, cfg.spect_start + cfg.d_spect), (cfg.spect_end - cfg.d_spect, cfg.spect_end)):
                ax.axvspan(a, b, color='c', alpha=0.4)
            ax.plot(np.arange(cfg.spect_start, cfg.spect_end + 1), res.spect_ave, 'r', lw=1,
                    label='after baseline (summed for the normalisation)')
            ax.set_xlabel('Pixel index')
            ax.legend(fontsize=7)
        ax.set_title(f'Spectrometer, file {res.fnr} (cyan: baseline windows)', fontsize=9)
        ax = axs[1]
        if res.bg_zoom is not None:
            z = res.bg_zoom
            im = ax.imshow(z, cmap='jet', origin='upper', extent=(0.5, z.shape[1] + 0.5, z.shape[0] + 0.5, 0.5))
            fig.colorbar(im, ax=ax, fraction=0.046, label='counts / pixel')
            ax.add_patch(Ellipse((res.cbar, res.rbar), cfg.ellipse[0], cfg.ellipse[1], fill=False, ec='w', lw=2))
        ax.set_title(f'Background subtracted from file {res.fnr} (ROI, mode {cfg.bg_mode})', fontsize=9)
        ax = axs[2]
        if stats:
            ax.plot([s['fnr'] for s in stats], [s['prbg'] for s in stats], '-o', ms=3, color='#a855f7')
        ax.set_xlabel('File number')
        ax.set_title('Proximity background per pixel (subtracted from the ROI)', fontsize=9)
        axs[3].set_visible(False)
        self._bg_axes = axs
        fig.tight_layout()
        self.canvas_bg.draw_idle()

    def plot_bg_check(self, bg, cfg):
        """MATLAB figures 23/24: background stability over the background files."""
        axs = getattr(self, '_bg_axes', None)
        if not axs or self.ana is None:
            return
        ax = axs[3]
        ax.clear()
        if bg is not None and len(bg[1]):
            ax.set_visible(True)
            ax.plot(bg[0], bg[1], '-o', color='#0ea5e9', ms=4)
            ax.set_xlabel('Background file delay')
            ax.set_title('Background variation (ROI·ellipse sum / I0 vs first BG file; inset: last - first)',
                         fontsize=9)
            z = self.ana.bg_check_zoom
            if z is not None:
                ins = ax.inset_axes([0.72, 0.6, 0.27, 0.38])
                im = ins.imshow(z, cmap='jet')
                ins.set_xticks([]); ins.set_yticks([])
                cb = self.fig_bg.colorbar(im, cax=ins.inset_axes([1.03, 0, 0.06, 1]))   # cleared with ax
                cb.ax.tick_params(labelsize=6)
        self.fig_bg.tight_layout()
        self.canvas_bg.draw_idle()

    # ---- other windows -------------------------------------------------
    def open_timing(self):
        TimingWindow(self.root)

    def open_trace(self):
        path = (self.last_output + '_nor.txt') if self.last_output else ''
        TraceWindow(self.root, path)

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


# --------------------------------------------------------------------------
# timing window (Svirgo)
# --------------------------------------------------------------------------

class TimingWindow:
    def __init__(self, master):
        self.win = tk.Toplevel(master)
        self.win.title('Timing pump A / pump B (Svirgo)')
        self.win.configure(background=PALETTE['bg'])
        header(self.win, 'Timing  ·  pump A vs pump B', 'Svirgo cross-correlation')
        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None
        self.trace_a = self.trace_b = None
        self._build()
        self.win.after(200, self._poll)
        self.win.protocol('WM_DELETE_WINDOW', self.close)

    def _build(self):
        left = ttk.Frame(self.win, padding=4)
        left.pack(side='left', fill='y')
        right = ttk.Frame(self.win, padding=4)
        right.pack(side='left', fill='both', expand=True)

        self.path_a = tk.StringVar()
        self.path_b = tk.StringVar()
        for r, (lab, var) in enumerate((('Pump A scan folder', self.path_a), ('Pump B scan folder', self.path_b))):
            ttk.Label(left, text=lab).grid(row=r, column=0, sticky='w')
            ttk.Entry(left, textvariable=var, width=34).grid(row=r, column=1, sticky='we')
            ttk.Button(left, text='…', width=3, command=lambda v=var: self._browse(v)).grid(row=r, column=2)
        r = 2
        self.f_pump = Field(left, r, 'Pump trajectory (1-4)', 4); r += 1
        self.f_thr = Field(left, r, 'FEL-off threshold', 10)
        ttk.Button(left, text='Auto', width=5, command=self.auto_thr).grid(row=r, column=2); r += 1

        self.n_rois = tk.IntVar(value=1)
        rf = ttk.Frame(left)
        rf.grid(row=r, column=0, columnspan=3, sticky='w'); r += 1
        ttk.Label(rf, text='Number of ROIs:').pack(side='left')
        ttk.Radiobutton(rf, text='1', variable=self.n_rois, value=1, command=self._roi_state).pack(side='left')
        ttk.Radiobutton(rf, text='2 (Wollaston prism)', variable=self.n_rois, value=2,
                        command=self._roi_state).pack(side='left')
        self.f_roi1 = Field(left, r, 'ROI 1 col, row, w, h', '75, 60, 70, 70', 18); r += 1
        self.f_roi2 = Field(left, r, 'ROI 2 col, row, w, h', '75, 60, 70, 70', 18); r += 1
        ttk.Button(left, text='Draw ROIs on the image…', command=self.draw_rois).grid(
            row=r, column=0, columnspan=3, sticky='we'); r += 1
        ttk.Button(left, text='Find the peak: automatic ROI', style='Accent.TButton', command=self.auto_roi).grid(
            row=r, column=0, columnspan=3, sticky='we', pady=2); r += 1
        self.corr_roi = tk.IntVar(value=2)
        self.rb_corr = ttk.Frame(left)
        self.rb_corr.grid(row=r, column=0, columnspan=3, sticky='w'); r += 1
        ttk.Label(self.rb_corr, text='Cross-correlate ROI:').pack(side='left')
        self.rb_c1 = ttk.Radiobutton(self.rb_corr, text='1', variable=self.corr_roi, value=1)
        self.rb_c1.pack(side='left')
        self.rb_c2 = ttk.Radiobutton(self.rb_corr, text='2', variable=self.corr_roi, value=2)
        self.rb_c2.pack(side='left')
        self.f_scale = Field(left, r, 'Plot scale A, B', '100, 100', 18); r += 1
        self.auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text='Auto update every (s)', variable=self.auto).grid(row=r, column=0, sticky='w')
        self.sleep = tk.StringVar(value='10')
        ttk.Entry(left, textvariable=self.sleep, width=6).grid(row=r, column=1, sticky='w'); r += 1
        bf = ttk.Frame(left)
        bf.grid(row=r, column=0, columnspan=3, sticky='we'); r += 1
        self.b_start = ttk.Button(bf, text='▶  Start', command=self.start, style='Run.TButton')
        self.b_start.pack(side='left', fill='x', expand=True)
        self.b_stop = ttk.Button(bf, text='■  Stop', command=self.stop_event.set, state='disabled',
                                 style='Stop.TButton')
        self.b_stop.pack(side='left', fill='x', expand=True)
        ttk.Button(left, text='Save pictures and traces…', style='Tool.TButton', command=self.save).grid(
            row=r, column=0, columnspan=3, sticky='we', pady=2); r += 1
        self.roi_fig = None
        self._contrast_cache = {}
        self.result = tk.StringVar(value='')
        ttk.Label(left, textvariable=self.result, wraplength=320, foreground=PALETTE['accent_dark'],
                  font=('Helvetica', 11, 'bold')).grid(
            row=r, column=0, columnspan=3, sticky='w', pady=6); r += 1
        self.log_box = console(left, height=10, width=46)
        self.log_box.grid(row=r, column=0, columnspan=3, sticky='nsew')
        self._roi_state()

        self.fig = Figure(figsize=(9, 8), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _browse(self, var):
        d = filedialog.askdirectory(parent=self.win)
        if d:
            var.set(d)

    def _roi_state(self):
        two = self.n_rois.get() == 2
        self.f_roi2.entry.configure(state='normal' if two else 'disabled')
        for w in (self.rb_c1, self.rb_c2):
            w.configure(state='normal' if two else 'disabled')

    def rois(self):
        r = [tuple(parse_list(self.f_roi1.get(), int))]
        if self.n_rois.get() == 2:
            r.append(tuple(parse_list(self.f_roi2.get(), int)))
        return r

    def _first_path(self):
        for p in (self.path_a.get(), self.path_b.get()):
            if p and tt.list_h5(p):
                return p
        raise FileNotFoundError('No .h5 files in pump A or pump B rawdata folder')

    def auto_thr(self):
        try:
            self.f_thr.set(f'{tt.auto_threshold(self._first_path()):.4g}')
        except Exception as e:
            messagebox.showerror('Threshold', str(e), parent=self.win)

    def _contrast(self):
        """FEL-on minus FEL-off image of the strongest files (cached per folder and threshold)."""
        path, thr = self._first_path(), float(self.f_thr.get())
        key = (path, thr, len(tt.list_h5(path)))
        if key not in self._contrast_cache:
            self.win.configure(cursor='watch')
            self.win.update_idletasks()
            try:
                self._contrast_cache[key] = tt.contrast_image(path, thr, log=self._log)[0]
            finally:
                self.win.configure(cursor='')
        return self._contrast_cache[key]

    def _log(self, m):
        self.log_box.insert('end', m + '\n')
        self.log_box.see('end')

    def auto_roi(self):
        """Locate the FEL spot(s) and set the ROI(s); show the image and the x / y cuts."""
        try:
            img = self._contrast()
        except Exception as e:
            messagebox.showerror('Automatic ROI', str(e), parent=self.win)
            return
        n = self.n_rois.get()
        found, work = [], img.copy()
        for k in range(n):
            a = tt.auto_roi(work)
            found.append(a)
            c, r, w, h = a['roi']
            work[r - 1:r - 1 + h, c - 1:c - 1 + w] = 0      # second spot (Wollaston): mask the first
        for f, a in zip((self.f_roi1, self.f_roi2), found):
            f.set(fmt_list(a['roi']))
            self._log(f"peak at col {a['peak'][0]}, row {a['peak'][1]} -> ROI {fmt_list(a['roi'])} "
                      f"(edges where the spot falls to {abs(a['level']):.3g}, {a['n_cuts']} cuts)")
        self.roi_fig = timing_roi_figure(img, found)
        figure_window(self.win, 'Automatic ROI', [self.roi_fig])

    def save(self):
        d = filedialog.askdirectory(parent=self.win, title='Folder for the timing pictures and traces')
        if not d:
            return
        saved = []
        self.fig.savefig(os.path.join(d, 'timing_traces.png'), dpi=150, bbox_inches='tight')
        saved.append('timing_traces.png')
        if self.roi_fig is not None:
            self.roi_fig.savefig(os.path.join(d, 'timing_roi.png'), dpi=150, bbox_inches='tight')
            saved.append('timing_roi.png')
        for tr, name in ((self.trace_a, 'A'), (self.trace_b, 'B')):
            if tr is None or not tr.t:
                continue
            t, i = tr.arrays()
            fn = f'timing_pump{name}.txt'
            np.savetxt(os.path.join(d, fn), np.column_stack([t] + [v * 100 for v in i]), fmt='%.7e', delimiter='\t',
                       header=f'{tr.path}\ndelay line (ps)\t' + '\t'.join(f'ROI {k + 1} variation (%)' for k in range(len(i))))
            saved.append(fn)
        self._log('saved ' + ', '.join(saved) + ' in ' + d)

    def draw_rois(self):
        try:
            img = self._contrast()
            title = 'FEL on - off'
        except Exception:
            try:
                img = tt.mean_image(self._first_path())
                title = 'mean image'
            except Exception as e:
                messagebox.showerror('ROI', str(e), parent=self.win)
                return
        n = self.n_rois.get()
        fig = Figure(figsize=(7, 6), dpi=90)
        ax = fig.add_subplot(111)
        im = ax.imshow(img, origin='upper', cmap='jet', extent=(0.5, img.shape[1] + 0.5, img.shape[0] + 0.5, 0.5))
        fig.colorbar(im, ax=ax, fraction=0.046, label='counts / pixel')
        ax.set_xlabel(title)
        ax.set_title(f'Drag ROI 1{" then ROI 2" if n == 2 else ""}; close the window when done')
        win = figure_window(self.win, 'Draw ROIs', [fig])
        fields = [self.f_roi1, self.f_roi2][:n]
        state = {'k': 0}

        def on_select(e1, e2):
            k = state['k'] % n
            x0, x1 = sorted((e1.xdata, e2.xdata))
            y0, y1 = sorted((e1.ydata, e2.ydata))
            roi = (int(round(x0)), int(round(y0)), max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))
            fields[k].set(fmt_list(roi))
            ax.add_patch(Rectangle((roi[0], roi[1]), roi[2], roi[3], fill=False, ec='rw'[k], lw=2))
            fig.canvas.draw_idle()
            state['k'] += 1

        win.selector = RectangleSelector(ax, on_select, useblit=False, interactive=False)

    def start(self):
        try:
            rois = self.rois()
            pump = int(self.f_pump.get())
            thr = float(self.f_thr.get())
            sleep = float(self.sleep.get())
            if any(len(r) != 4 for r in rois) or not 1 <= pump <= 4:
                raise ValueError('ROIs need 4 numbers, pump trajectory 1-4')
        except ValueError as e:
            messagebox.showerror('Timing', str(e), parent=self.win)
            return
        self.trace_a = tt.PumpTrace(self.path_a.get())
        self.trace_b = tt.PumpTrace(self.path_b.get())
        self.stop_event.clear()
        self.b_start.configure(state='disabled')
        self.b_stop.configure(state='normal')
        self.result.set('')
        args = (rois, pump, thr, sleep, self.auto.get(), self.corr_roi.get() if len(rois) == 2 else 1)
        self.worker = threading.Thread(target=self._work, args=args, daemon=True)
        self.worker.start()

    def _work(self, rois, pump, thr, sleep, auto, corr_roi):
        log = lambda m: self.q.put(('log', m))
        try:
            while True:
                for tr, name in ((self.trace_a, 'A'), (self.trace_b, 'B')):
                    if not tr.path or not tt.list_h5(tr.path):
                        log(f'No pump {name} data ready')
                        continue
                    tr.update(rois, thr, log=log)
                self.q.put(('plot', len(rois)))
                na, nb = len(self.trace_a.t), len(self.trace_b.t)
                if na == nb and nb > 0:
                    res = tt.compare(self.trace_a, self.trace_b, pump, corr_roi - 1)
                    self.q.put(('result', res))
                    break
                if not auto or self.stop_event.is_set():
                    break
                for _ in range(int(sleep * 10)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.1)
                if self.stop_event.is_set():
                    break
        except Exception as e:
            self.q.put(('error', f'{type(e).__name__}: {e}', traceback.format_exc()))
        finally:
            self.q.put(('done',))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == 'log':
                    self.log_box.insert('end', msg[1] + '\n')
                    self.log_box.see('end')
                elif msg[0] == 'plot':
                    self.plot(msg[1])
                elif msg[0] == 'result':
                    self.plot(len(self.trace_a.i), msg[1])
                    self.result.set(msg[1].message)
                    self.log_box.insert('end', msg[1].message + '\n')
                elif msg[0] == 'error':
                    self.log_box.insert('end', msg[2] + '\n')
                    messagebox.showerror('Timing', msg[1], parent=self.win)
                elif msg[0] == 'done':
                    self.b_start.configure(state='normal')
                    self.b_stop.configure(state='disabled')
        except queue.Empty:
            pass
        if self.win.winfo_exists():
            self.win.after(200, self._poll)

    def plot(self, n_rois, res=None):
        try:
            sa, sb = parse_list(self.f_scale.get(), float)[:2]
        except ValueError:
            sa = sb = 100.0
        ta_, ia = self.trace_a.arrays()
        tb_, ib = self.trace_b.arrays()
        self.fig.clear()
        rows = n_rois + (1 if res is not None else 0)
        for k in range(n_rois):
            ax = self.fig.add_subplot(rows, 1, k + 1)
            if ta_.size and len(ia) > k:
                ax.plot(ta_, ia[k] * sa, 'r-o', lw=2, label=os.path.basename(os.path.dirname(
                    os.path.normpath(self.trace_a.path))) or 'A')
            if tb_.size and len(ib) > k:
                ax.plot(tb_, ib[k] * sb, '-o', color='#4f46e5', lw=2, label=os.path.basename(os.path.dirname(
                    os.path.normpath(self.trace_b.path))) or 'B')
            ax.grid(True, which='both')
            ax.set_xlabel('Delay line position (ps)')
            ax.set_ylabel(f'Norm. ROI {k + 1} variation (%)')
            try:
                ax.set_title(f'ROI {k + 1}: col, row, w, h = {fmt_list(self.rois()[k])}')
            except (ValueError, IndexError):
                pass
            if ax.lines:
                ax.legend(loc='best')
        if res is not None:
            ax = self.fig.add_subplot(rows, 1, rows)
            ax.stem(res.lags, res.xc)
            ax.set_xlabel('Lag (points)')
            ax.set_title(f'Cross-correlation: dT(A-B) = {res.time_diff:.3f} ps')
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def close(self):
        self.stop_event.set()
        self.win.destroy()


# --------------------------------------------------------------------------
# trace analysis window
# --------------------------------------------------------------------------

class ParamTable:
    """Editable p0 / lower / upper table for a list of parameter names."""

    def __init__(self, parent, names, p0, cons):
        self.frame = ttk.Frame(parent)
        self.names = list(names)
        ttk.Label(self.frame, text='').grid(row=0, column=0)
        for j, h in enumerate(('start', 'lower', 'upper')):
            ttk.Label(self.frame, text=h, foreground=PALETTE['accent'], font=('Helvetica', 10, 'bold')).grid(
                row=0, column=j + 1)
        self.vars = []
        for i, n in enumerate(self.names):
            ttk.Label(self.frame, text=n, font=('Helvetica', 10, 'bold')).grid(row=i + 1, column=0, sticky='e', padx=2)
            row = []
            for j, v in enumerate((p0[i], cons[0][i], cons[1][i])):
                var = tk.StringVar(value='' if v is None else f'{v:.6g}')
                ttk.Entry(self.frame, textvariable=var, width=10).grid(row=i + 1, column=j + 1)
                row.append(var)
            self.vars.append(row)

    def get(self):
        vals = np.array([[float(v.get()) if v.get().strip() else np.nan for v in row] for row in self.vars])
        return tuple(vals[:, 0]), (tuple(vals[:, 1]), tuple(vals[:, 2]))


class TraceWindow:
    MODELS = [(0, 'Thermal only'), (1, '1 frequency (1 freq 1 dump)'),
              (2, '2 frequencies (2 freq 1 dump)'), (3, '3 frequencies (3 freq 2 dump)')]

    def __init__(self, master, path=''):
        self.master = master
        self.win = tk.Toplevel(master)
        self.win.title('Trace analysis')
        self.win.configure(background=PALETTE['bg'])
        header(self.win, 'Trace analysis  ·  thermal + FFT + global fits', "Den's notebook models")
        self.tables = []
        self.report_items = []   # (title, figures, table) of the last run, merge and overlay
        self._build(path)

    def _build(self, path):
        f = ttk.Frame(self.win, padding=6)
        f.pack(fill='both', expand=True)
        r = 0
        ttk.Label(f, text='Trace file (_nor.txt)').grid(row=r, column=0, sticky='w')
        self.path = tk.StringVar(value=path)
        ttk.Entry(f, textvariable=self.path, width=46).grid(row=r, column=1, columnspan=2, sticky='we')
        ttk.Button(f, text='…', width=3, command=self._browse).grid(row=r, column=3); r += 1

        sc = ttk.LabelFrame(f, text='Several scans: merge repeated scans, or overlay scans / samples', padding=4)
        sc.grid(row=r, column=4, rowspan=6, sticky='nsew', padx=(8, 0))   # beside the trace settings
        self.scan_list = tk.Listbox(sc, height=5, width=44, selectmode='extended',
                                    background=PALETTE.get('field', 'white'))
        self.scan_list.grid(row=0, column=0, rowspan=4, sticky='nsew', padx=2)
        for k, (text, cmd) in enumerate((('Add files…', self._add_scans), ('Add this trace', self._add_current),
                                         ('Remove', self._remove_scans), ('Clear', lambda: self.scan_list.delete(0, 'end')))):
            ttk.Button(sc, text=text, command=cmd).grid(row=k, column=1, sticky='we', padx=2, pady=1)
        self.scale_scans = tk.BooleanVar(value=False)
        ttk.Checkbutton(sc, text='Merge: scale each scan to the first one (least squares) before averaging',
                        variable=self.scale_scans).grid(row=4, column=0, columnspan=2, sticky='w')
        self.ov_norm = tk.BooleanVar(value=True)
        ttk.Checkbutton(sc, text='Overlay: normalise (0 = baseline below, 1 = peak)',
                        variable=self.ov_norm).grid(row=5, column=0, columnspan=2, sticky='w')
        self.f_ov_base = Field(sc, 6, 'Overlay baseline: first N points (one N, or N1, N2, …; last repeats)', '5', 8)
        self.f_offset = Field(sc, 7, 'Overlay: vertical offset between traces', '0', 8)
        sb = ttk.Frame(sc)
        sb.grid(row=8, column=0, columnspan=2, sticky='we', pady=2)
        ttk.Button(sb, text='⊕  Merge (weighted average) → fit it', command=self.merge,
                   style='Accent.TButton').pack(side='left', fill='x', expand=True, padx=2)
        ttk.Button(sb, text='≋  Overlay', command=self.overlay, style='Tool.TButton').pack(
            side='left', fill='x', expand=True, padx=2)
        sc.columnconfigure(0, weight=1)

        self.f_cols = Field(f, r, 'Columns x, y (1-based)', '1, 2', 12); r += 1
        self.f_shift = Field(f, r, 'Time shift (ps), x = t - shift', '0', 12); r += 1
        self.f_base = Field(f, r, 'Baseline points from, to', '0, 5', 12); r += 1
        self.f_crop = Field(f, r, 'Use points start, stop', '0, ', 12); r += 1
        self.f_cut = Field(f, r, 'FFT: skip first points (cut)', '0', 12); r += 1
        self.f_xlim = Field(f, r, 'FFT plot limit = highest ν searched (GHz)', '120', 12); r += 1

        th = ttk.LabelFrame(f, text='Thermal decay  |0.5(1+erf((t-t0)/σ))(a_exp e^-(t-t0)/τ + y0)|²', padding=4)
        th.grid(row=r, column=0, columnspan=5, sticky='we', pady=4); r += 1
        tr = ttk.Frame(th)
        tr.pack()
        self.thermal_table = ParamTable(tr, tf.THERMAL_NAMES, tf.THERMAL_P0, tf.THERMAL_CONS)
        self.thermal_table.frame.grid(row=0, column=0, sticky='n')
        elf = ttk.LabelFrame(tr, text='Coherent spike / electronic peak', padding=4)
        elf.grid(row=0, column=1, sticky='n', padx=(16, 0))
        self.electronic = tk.BooleanVar(value=False)
        ttk.Checkbutton(elf, text='Fit a fast term at t0 (in all fits)', variable=self.electronic,
                        style='Head.TCheckbutton').pack(anchor='w')
        self.el_mode = tk.StringVar(value=tf.EL_MODE)
        for mode, text in (('cs', 'Coherent spike:  |I_TG ± B·G|² + |C·G|²,  G = exp(-((t-t_cs)/σ_cs)²)'),
                           ('free', 'Free peak:  |I_TG + a_el·(step at t_el, rise σ_el, decay τ_el)|²')):
            ttk.Radiobutton(elf, text=text, value=mode, variable=self.el_mode,
                            command=self._el_mode_changed).pack(anchor='w')
        self.el_box = ttk.Frame(elf)
        self.el_box.pack(anchor='w')
        self._el_mode_changed()
        ttk.Label(elf, text='blank start = from the data (height and steepest rise of the spike)',
                  font=('Helvetica', 9)).pack(anchor='w')
        skf = ttk.Frame(elf)
        skf.pack(anchor='w', pady=(4, 0))
        self.f_skip = Field(skf, 0, 'Thermal fit skips the spike, t from, to (ps; blank = auto)', '', 12)
        self.fix_thermal = tk.BooleanVar(value=True)
        ttk.Checkbutton(elf, text='Frequency fits keep t0, σ, a_exp, τ, y0 of the thermal fit (fixed)',
                        variable=self.fix_thermal).pack(anchor='w')
        flt = ttk.Frame(th)
        flt.pack(fill='x')
        self.f_band = Field(flt, 0, 'Filter: reject if τ±err band > (% of signal)', '100', 6)
        self.f_r2 = Field(flt, 1, 'Filter: reject if R² <', '0.3', 6)
        self.show_rejected = tk.BooleanVar(value=False)
        ttk.Checkbutton(flt, text='Plot the thermal fit even when rejected', variable=self.show_rejected).grid(
            row=2, column=0, columnspan=2, sticky='w')

        md = ttk.LabelFrame(f, text='Fits to run', padding=4)
        md.grid(row=r, column=0, columnspan=5, sticky='we'); r += 1
        self.model_vars = {}
        for k, (n, label) in enumerate(self.MODELS):
            v = tk.BooleanVar(value=(n == 0))
            ttk.Checkbutton(md, text=label, variable=v).grid(row=k // 2, column=k % 2, sticky='w', padx=4)
            self.model_vars[n] = v
        ttk.Label(md, text='Always shown: trace + thermal decay, FFT of the residual + lorentzian fit.\n'
                           'Several frequency boxes ticked: fits and residual FFTs overlaid for comparison.').grid(
            row=2, column=0, columnspan=2, sticky='w', padx=4)
        self.plot_dt = tk.BooleanVar(value=True)
        ttk.Checkbutton(md, text='Show τ ± error band', variable=self.plot_dt).grid(
            row=3, column=0, columnspan=2, sticky='w', padx=4)
        self.f_nu = Field(md, 4, 'Frequency guesses ν1, ν2, ν3 (GHz, blank = from FFT)', '', 14)
        self.f_nuspan = Field(md, 5, 'Frequency span ±Δν1, Δν2, Δν3 (GHz, blank = ±3)', '', 14)
        self.f_numin = Field(md, 6, 'Ignore FFT peaks below (GHz)', '5', 14)
        self.weighted = tk.BooleanVar(value=False)
        ttk.Checkbutton(md, text='Weighted fit: points weighted by 1/error² (error bars from _err.txt); '
                                 'unticked = unweighted, as the notebook', variable=self.weighted,
                        style='Head.TCheckbutton').grid(row=7, column=0, columnspan=2, sticky='w', padx=4)

        bf = ttk.Frame(f)
        bf.grid(row=r, column=0, columnspan=5, sticky='we', pady=4); r += 1
        ttk.Button(bf, text='▶  Run fits', command=self.run, style='Run.TButton').pack(
            side='left', fill='x', expand=True, padx=2)
        ttk.Button(bf, text='✎  Edit guesses and refit…', command=self.edit_and_refit, style='Accent.TButton').pack(
            side='left', fill='x', expand=True, padx=2)
        ttk.Button(bf, text='Save tables…', command=self.save_tables).pack(side='left', fill='x', expand=True, padx=2)
        ttk.Button(bf, text='📄  PDF report…', command=self.report).pack(side='left', fill='x', expand=True, padx=2)
        self.out = console(f, height=12, width=100)
        self.out.configure(wrap='none')
        self.out.grid(row=r, column=0, columnspan=5, sticky='nsew')
        f.columnconfigure(1, weight=1)
        f.rowconfigure(r, weight=1)
        self.last_guess = {}

    def _browse(self):
        p = filedialog.askopenfilename(parent=self.win, filetypes=[('Text', '*.txt *.dat'), ('All', '*')])
        if p:
            self.path.set(p)

    def _say(self, text):
        self.out.insert('end', text + '\n')
        self.out.see('end')
        self.tables.append(text)

    def _load(self):
        data = np.loadtxt(self.path.get())
        cx, cy = parse_list(self.f_cols.get(), int)[:2]
        base = parse_list(self.f_base.get(), int)
        crop = [t.strip() for t in self.f_crop.get().split(',')] + ['', '']
        start = int(crop[0]) if crop[0] else 0
        stop = int(crop[1]) if crop[1] else None
        x, y = tf.prepare_trace(data[:, cx - 1], data[:, cy - 1], float(self.f_shift.get() or 0),
                                base[0], base[1], start, stop)
        return x, y, self._load_err(data[:, cx - 1], data[:, cy - 1], start, stop)

    def _load_err(self, time, signal, start, stop, path=None):
        """Propagated error from Scan_NNN_err.txt next to the _nor file, scaled and
        ordered like prepare_trace does with the signal. None when there is no file."""
        p = path or self.path.get()
        ep = p[:-len('_nor.txt')] + '_err.txt' if p.endswith('_nor.txt') else None
        if not ep or not os.path.exists(ep):
            return None
        try:
            d = np.atleast_2d(np.loadtxt(ep))
        except (OSError, ValueError):
            return None
        if len(d) != len(time) or not np.allclose(d[:, 0], time):
            return None
        signal = np.asarray(signal, float)
        e = d[:, 2] / (signal.max() - signal.min())
        e = e[np.argsort(np.asarray(time, float), kind='stable')]
        return e[start:stop]

    def _el_mode_changed(self):
        """Coherent spike (B and C, shares t0 and σ) or free peak (4 parameters)."""
        tf.set_el_mode(self.el_mode.get())
        for w in self.el_box.winfo_children():
            w.destroy()
        auto = {'B', 'C', 't_cs', 'a_el', 't_el'}
        self.el_table = ParamTable(self.el_box, tf.EL_NAMES,
                                   tuple(None if n in auto else v for n, v in zip(tf.EL_NAMES, tf.EL_P0)), tf.EL_CONS)
        self.el_table.frame.pack(anchor='w')

    def _show(self, title, figs, table=None):
        figure_window(self.win, title, figs, table)
        self.report_items.append((title, figs, table))

    def run(self, override=None):
        """Always: trace + thermal decay, FFT of the residual + lorentzian fit.
        Then each ticked frequency model; one model -> the notebook figures,
        several -> fits and residual FFTs overlaid in the same axes."""
        self.report_items = [it for it in self.report_items if it[0].startswith(('Merged', 'Overlay'))]
        try:
            x, y, e = self._load()
            cut = int(self.f_cut.get() or 0)
            xlim = float(self.f_xlim.get() or 120)
            p0, cons = self.thermal_table.get()
            el = self.electronic.get()
            keep = np.ones(len(x), bool)
            if el:   # the thermal fit leaves the spike out; the fast term enters only the frequency fits
                tf.set_el_mode(self.el_mode.get())
                ep0, econs = self.el_table.get()
                auto = tf.guess_el(x, y)
                ep0 = tuple(a if np.isnan(v) else v for v, a in zip(ep0, auto))
                sk = parse_list(self.f_skip.get(), float) if self.f_skip.get() else []
                win = tuple(sk[:2]) if len(sk) >= 2 else tf.spike_window(x, y)
                keep = (x < win[0]) | (x > win[1])
            w_err = e if self.weighted.get() else None
            if self.weighted.get() and e is None:
                self._say('Weighted fit asked, but there is no _err.txt for this trace: fits are unweighted')
            xk, yk, ek = x[keep], y[keep], (None if w_err is None else w_err[keep])
            with tf.weighted(xk, ek):
                th = tf.fit_exp(xk, yk, p0, cons)
            ok, reasons, q = tf.thermal_quality(xk, yk, th, cons, max_band=float(self.f_band.get() or 100)/100,
                                                min_r2=float(self.f_r2.get() or 0))
            res_k = yk - th['fit']
            th['fit'] = tf.thermal_model(x, *th['popt'][:5])   # on every point, for the residual and FFT
            y_fft = y.copy()
            if el:   # no spike in the residual that goes to the FFT
                y_fft[~keep] = th['fit'][~keep]
                for fig in th['figures']:
                    for ax in fig.axes:
                        ax.axvspan(*win, color='gold', alpha=0.3, lw=0)
                        ax.plot(x[~keep], y[~keep], 'x', color='darkorange', ms=7, mew=2)
        except Exception as ex:
            messagebox.showerror('Trace analysis', f'{type(ex).__name__}: {ex}', parent=self.win)
            return
        steps = tf.time_steps(x[cut:])
        if len(steps) > 1:
            self._say('Time steps differ along the scan: ' + ', '.join(
                f'{st:.3g} ps from {a:.4g} to {b:.4g} ({n} pts)' for st, a, b, n in steps)
                + '\n-> FFTs computed on the real time points (not assuming one step)')
        verdict = 'accepted' if ok else 'REJECTED: ' + '; '.join(reasons)
        table = th['table'] + f'\nfilter: R² = {q["r2"]:.2f}, τ band {q["band"]*100:.0f}% -> {verdict}'
        if el:
            table += (f'\nspike left out of the thermal fit: t = {win[0]:.3g} to {win[1]:.3g} ps '
                      f'({int((~keep).sum())} points, orange)')
        table += '\n' + self._fit_note(res_k, None if e is None else e[keep], len(th['popt']))
        if ok or self.show_rejected.get():
            figs = th['figures']
            if not ok:
                figs[0].suptitle('Thermal fit rejected: ' + '; '.join(reasons), fontsize=16, color='darkred')
        else:
            figs = [tf.trace_only_figure(x, y, 'Thermal fit rejected, not plotted: ' + '; '.join(reasons))]
        add_data_band(figs, x, y, e)
        self._show('Trace + thermal decay' + ('' if ok else ' (fit rejected)'), figs, table)
        self._say('Thermal decay\n' + table)

        selected = [n for n in (1, 2, 3) if self.model_vars[n].get()]
        nu_guess = parse_list(self.f_nu.get(), float) if self.f_nu.get() else []
        nu_min = float(self.f_numin.get() or 0)
        spans = parse_list(self.f_nuspan.get(), float) if self.f_nuspan.get() else []
        n_lor = max(selected + [len(nu_guess), 1])
        try:
            ff, fr = tf.residual_fft(x, y_fft, th['fit'], cut)
            nus = tf.find_fft_peaks(ff, fr, n_lor, nu_guess, nu_min, xlim)
            lp0, lcons = tf.guess_lorentz(ff, fr, nus, spans)
            lr = tf.DO_FFT_LORENTZ[n_lor](x, y_fft, th['fit'], lp0, lcons, cut, xlim)
        except Exception as ex:
            self._say(f'FFT lorentzian fit failed: {type(ex).__name__}: {ex}')
            return
        if not ok:
            lr['figures'][0].suptitle('Thermal fit rejected: this residual (and its FFT) may be distorted',
                                      fontsize=16, color='darkred')
        self._show('FFT of the thermal residual + lorentzian fit', lr['figures'], lr['table'])
        self._say(lr['title'] + '\n' + lr['table'])

        results = {}
        for n in selected:
            try:
                n_par = len((tf.GLOBAL_MODELS_EL if el else tf.GLOBAL_MODELS)[n][1])
                if override and n in override and len(override[n][0]) == n_par:
                    gp0, gcons = override[n][:2]
                else:   # no edited guesses, or they were made with the electronic peak toggled the other way
                    th_p, th_c = th['popt'], cons
                    if el:   # the fast term starts from its own table (blank = from the data)
                        th_p = np.r_[th['popt'][:5], ep0]
                        th_c = (tuple(cons[0]) + tuple(econs[0]), tuple(cons[1]) + tuple(econs[1]))
                    gp0, gcons = tf.guess_global(n, th_p, th_c, lr['popt'], ff[1] - ff[0], spans, nus, el)
                    if el and self.fix_thermal.get():   # the thermal part is the spike-free thermal fit
                        gp0, gcons = tf.fix_params(gp0, gcons, tf.GLOBAL_MODELS_EL[n][1],
                                                   dict(zip(tf.THERMAL_NAMES, th['popt'][:5])))
                self.last_guess[n] = (gp0, gcons, lr)
                with tf.weighted(x, w_err):
                    g = tf.run_global(n, x, y, lr['fft_res'], lr['fit_lor_tot'], cut, xlim, gp0, gcons,
                                      1 if self.plot_dt.get() else 0, el)
                g['table'] += '\n' + self._fit_note(g['res'], e, len(g['popt']))
                self._say(g['title'] + '\n' + g['table'])
                self.last_guess[n] = (gp0, gcons, lr, g['popt'])
                results[n] = g
            except Exception as ex:
                self._say(f'{n} frequency fit failed: {type(ex).__name__}: {ex}')
        if len(results) == 1:
            g = next(iter(results.values()))
            add_data_band(g['figures'], x, y, e)
            self._show(g['title'], g['figures'], g['table'])
        elif len(results) > 1:
            cfigs = tf.compare_fits(x, y, results, xlim)
            add_data_band(cfigs, x, y, e)
            self._show('Comparison: ' + ', '.join(g['title'] for g in results.values()), cfigs,
                       '\n\n'.join(g['title'] + '\n' + g['table'] for g in results.values()))

    def _fit_note(self, res, e, n_par):
        """How the fit was weighted, and its reduced χ² when there are error bars."""
        how = 'weighted fit (1/error²)' if self.weighted.get() and e is not None else 'unweighted fit'
        if e is None:
            return how + ' (no _err.txt: no χ²)'
        return f'{how}; reduced χ² = {tf.chi2_red(res, e, n_par):.2f} ({len(res)} points, {n_par} parameters; ' \
               f'≈1 = the curve is as good as the error bars allow)'

    # ---- several scans ----------------------------------------------------
    def _add_scans(self):
        # plain extensions only: macOS Tk aborts Python on a pattern like '*_nor.txt'
        ps = filedialog.askopenfilenames(parent=self.win, title='Scan_NNN_nor.txt files',
                                         filetypes=[('Text', '*.txt *.dat'), ('All', '*')])
        for p in self.win.tk.splitlist(ps):
            if p.endswith('_err.txt'):   # the error file goes with its _nor trace, read automatically
                p = p[:-len('_err.txt')] + '_nor.txt'
            if p not in self.scan_list.get(0, 'end'):
                self.scan_list.insert('end', p)
        self.scan_list.xview_moveto(1.0)   # show the file names, not the start of the paths

    def _add_current(self):
        p = self.path.get().strip()
        if p and p not in self.scan_list.get(0, 'end'):
            self.scan_list.insert('end', p)

    def _remove_scans(self):
        for k in reversed(self.scan_list.curselection()):
            self.scan_list.delete(k)

    @staticmethod
    def _scan_label(p):
        return f'{os.path.basename(os.path.dirname(p))} {os.path.basename(p).replace("_nor.txt", "")}'.strip()

    def _scan_paths(self, n_min):
        ps = list(self.scan_list.get(0, 'end'))
        if len(ps) < n_min:
            messagebox.showinfo('Several scans', f'Add at least {n_min} _nor.txt files to the list first.',
                                parent=self.win)
            return None
        return ps

    def merge(self):
        """Weighted average of the listed scans, written as Merged_..._nor.txt and
        _err.txt in the folder of the first scan; it becomes the trace to fit."""
        ps = self._scan_paths(2)
        if not ps:
            return
        try:
            scans = [tf.read_scan(p) for p in ps]
            T, Y, E, S, N, fac = tf.merge_scans(scans, scale=self.scale_scans.get())
            stems = [os.path.basename(p).replace('_nor.txt', '') for p in ps]
            nums = [m.group(1) for m in (re.search(r'Scan_(\d+)', st) for st in stems) if m]
            name = ('Merged_Scan_' + '+'.join(nums)) if len(nums) == len(stems) else 'Merged_' + '+'.join(stems)
            out = os.path.join(os.path.dirname(ps[0]), name + '_nor.txt')
            tf.write_merged(out, T, Y, E, S, N, ps, fac)
        except Exception as ex:
            messagebox.showerror('Merge', f'{type(ex).__name__}: {ex}', parent=self.win)
            return
        weighted_ok = all(sc[2] is not None for sc in scans)
        fig = Figure(figsize=(12, 6), dpi=80)
        ax = fig.add_subplot(111)
        colors = ['#4f46e5', '#0ea5e9', '#10b981', '#f59e0b', '#a855f7', '#f97316', '#14b8a6', '#ec4899']
        for k, ((t, y, e), f) in enumerate(zip(scans, fac)):
            error_band(ax, t, y * f, (e if e is not None else np.zeros_like(y)) * abs(f), colors[k % len(colors)],
                       ms=2, label=self._scan_label(ps[k]) + (f' × {f:.3g}' if f != 1 else ''))
        error_band(ax, T, Y, np.nan_to_num(E), '#e11d48', ms=4, label='merged')
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('TG signal (counts per shot)')
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_title(f'{name}: ' + ('weighted average (1/error²)' if weighted_ok else
                                     'plain average (some scans have no _err.txt)'))
        text = (f'written {out}\nand {out[:-len("_nor.txt")]}_err.txt\n'
                + '\n'.join(f'  {p}  × {f:.4g}' for p, f in zip(ps, fac))
                + f'\n{len(T)} delays, measured by {N.min()}-{N.max()} scans')
        self.report_items = [it for it in self.report_items if not it[0].startswith('Merged')]
        self._show('Merged: ' + name, [fig], text)
        self._say(text)
        self.path.set(out)

    def overlay(self):
        """All listed traces in the same axes, with their error bands."""
        ps = self._scan_paths(1)
        if not ps:
            return
        norm = self.ov_norm.get()
        shift = float(self.f_shift.get() or 0)
        off = float(self.f_offset.get() or 0)
        n_base = [max(int(v), 0) for v in parse_list(self.f_ov_base.get(), float)] or [0]
        notes = []
        fig = Figure(figsize=(12, 6), dpi=80)
        ax = fig.add_subplot(111)
        colors = ['#e11d48', '#4f46e5', '#10b981', '#f59e0b', '#0ea5e9', '#a855f7', '#f97316', '#14b8a6']
        for k, p in enumerate(ps):
            try:
                t, y, e = tf.read_scan(p)
            except Exception as ex:
                self._say(f'{p}: {type(ex).__name__}: {ex}')
                continue
            e = np.zeros_like(y) if e is None else e
            x = t - shift
            if norm:
                # zero = mean of the first N points of this trace (N per trace, the last
                # value repeats), then peak = 1
                nb = n_base[min(k, len(n_base) - 1)]
                b0 = float(np.mean(y[:nb])) if nb > 0 else float(y.min())
                span = (y.max() - b0) or 1.0
                y, e = (y - b0) / span, e / span
                notes.append(f'{self._scan_label(p)}: baseline = ' +
                             (f'mean of the first {min(nb, len(y))} points (to {x[min(nb, len(y)) - 1]:.3g} ps)'
                              if nb > 0 else 'minimum'))
            error_band(ax, x, y + k * off, e, colors[k % len(colors)], ms=2, label=self._scan_label(p))
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('Δt [ps]')
        ax.set_ylabel('TG signal [a.u.] (normalised)' if norm else 'TG signal (counts per shot)')
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_title(f'{len(ps)} traces' + (f', offset {off:g}' if off else ''))
        self.report_items = [it for it in self.report_items if not it[0].startswith('Overlay')]
        self._show('Overlay of ' + ', '.join(self._scan_label(p) for p in ps), [fig], '\n'.join(notes) or None)

    def report(self):
        """PDF: settings of this window and every fit table, then all the figures
        of the last run (plus the last merge and overlay)."""
        if not self.report_items:
            messagebox.showinfo('PDF report', 'Run the fits (or a merge / overlay) first.', parent=self.win)
            return
        p0 = self.path.get()
        init = (os.path.basename(p0).replace('_nor.txt', '') or 'trace') + '_fits_report.pdf'
        path = filedialog.asksaveasfilename(parent=self.win, defaultextension='.pdf', initialfile=init,
                                            initialdir=os.path.dirname(p0) or None,
                                            filetypes=[('PDF', '*.pdf')])
        if not path:
            return
        fields = [v for v in vars(self).values() if isinstance(v, Field)]
        th_p0, th_cons = self.thermal_table.get()
        lines = [f'trace: {p0}', time.strftime('report written %Y-%m-%d %H:%M'), '']
        lines += [f'{fd.label}: {fd.get()}' for fd in fields]
        lines += [f'fits ticked: {[n for n, v in self.model_vars.items() if v.get()]}',
                  f'weighted fit: {self.weighted.get()}',
                  f'fast term: {self.electronic.get()} ({self.el_mode.get()}, thermal fixed: {self.fix_thermal.get()})' + (
                      ' (' + ', '.join(f'{n} = ' + ('auto' if np.isnan(v) else f'{v:g}') + f' [{lo:g}, {hi:g}]'
                                       for n, v, lo, hi in zip(tf.EL_NAMES, self.el_table.get()[0], *self.el_table.get()[1])) + ')'
                      if self.electronic.get() else ''),
                  f'scans listed: {list(self.scan_list.get(0, "end"))}',
                  'thermal guesses ' + ', '.join(f'{n} = {v:g} [{lo:g}, {hi:g}]' for n, v, lo, hi in
                                                 zip(tf.THERMAL_NAMES, th_p0, *th_cons)), '']
        text = '\n'.join(lines) + '\n'.join(f'\n--- {t} ---\n{tab}' for t, _, tab in self.report_items if tab)
        try:
            write_pdf(path, text_pages('Trace analysis report', text)
                      + [fig for _, figs, _ in self.report_items for fig in figs])
        except Exception as ex:
            messagebox.showerror('PDF report', f'{type(ex).__name__}: {ex}', parent=self.win)
            return
        self._say(f'report written: {path}')

    def edit_and_refit(self):
        if not self.last_guess:
            messagebox.showinfo('Trace analysis', 'Run the fits once first; the guesses then become editable.',
                                parent=self.win)
            return
        win = tk.Toplevel(self.win)
        win.title('Edit guesses')
        win.configure(background=PALETTE['bg'])
        tables = {}
        for n, item in sorted(self.last_guess.items()):
            gp0, gcons = item[0], item[1]
            if len(item) > 3:
                gp0 = tf._clip(item[3], *gcons)  # start from the last fitted values
            model = tf.GLOBAL_MODELS_EL[n] if len(gp0) == len(tf.GLOBAL_MODELS_EL[n][1]) else tf.GLOBAL_MODELS[n]
            lf = ttk.LabelFrame(win, text=model[0].__name__, padding=4)
            lf.pack(side='left', fill='y', padx=4)
            t = ParamTable(lf, model[1], gp0, gcons)
            t.frame.pack()
            tables[n] = (t, item[2])

        def refit():
            try:
                override = {n: (*t.get(), lr) for n, (t, lr) in tables.items()}
            except ValueError as e:
                messagebox.showerror('Edit guesses', str(e), parent=win)
                return
            for n in (1, 2, 3):
                self.model_vars[n].set(n in override)
            self.run(override)
        ttk.Button(win, text='▶  Refit', command=refit, style='Run.TButton').pack(side='bottom', fill='x')

    def save_tables(self):
        p = filedialog.asksaveasfilename(parent=self.win, defaultextension='.txt',
                                         initialfile=os.path.splitext(os.path.basename(self.path.get()))[0] + '_fits.txt')
        if p:
            with open(p, 'w', encoding='utf-8') as fh:
                fh.write('\n\n'.join(self.tables))


def main():
    root = tk.Tk()
    TimerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
