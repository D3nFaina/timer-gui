"""Post-analysis of a finished TG trace: thermal decay, FFT of the residual and
global fits with 1, 2 or 3 acoustic frequencies.

The fit models and figures are Den's notebook functions from
SiO2_Baldi_Analysis.ipynb, kept as they are:

    notebook                      here
    --------                      ----
    fit_exp                       fit_exp
    do_fft_lorentz_2 / _3         do_fft_lorentz_2 / _3  (+ do_fft_lorentz_1)
    freq1_dump1   (1 freq 1 dump) freq1_dump1
    freq2_dump1   (2 freq 1 dump) freq2_dump1
    freq3_dump2   (3 freq 2 dump, "2 e 3")  freq3_dump2

Changes, all mechanical:
- curve_fit is a thin wrapper: inside `weighted(x, err)` the fits to the
  trace use sigma = err (weighted fit); outside it they are the notebook's.
- p0 and cons are arguments instead of notebook globals.
- Figures are matplotlib Figure objects (no pyplot), so the GUI can show them
  in its own windows; each function returns them in a dict together with the
  fit results and the printed table.
- do_fft_lorentz_2 called lorentzian_3 with undefined x03/a3/gam3; it now
  uses lorentzian_2. do_fft_lorentz_1 is new, built like _2 and _3.
- guess_* helpers build p0/cons from the FFT peaks and the thermal fit.

Units as in the notebook: time in ps, frequencies in THz (shown x1000 as GHz).
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from scipy.fftpack import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit as _scipy_curve_fit
from scipy.signal import find_peaks
from scipy.special import erf, erfc, erfcx

try:
    from tabulate import tabulate
except ImportError:  # plain fallback, same information
    def tabulate(rows, headers, tablefmt=None):
        cols = [list(headers)] + [list(r) for r in rows]
        w = [max(len(str(c[i])) for c in cols) for i in range(len(headers))]
        line = lambda r: "| " + " | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)) + " |"
        return "\n".join([line(headers), "|-" + "-+-".join("-" * x for x in w) + "-|"]
                         + [line(r) for r in rows])

# --------------------------------------------------------------------------
# weighted fits: the notebook functions call curve_fit as written; while
# `weighted(x, err)` is active, every fit to that same x (thermal and global
# fits, not the lorentzians, which fit the FFT) gets sigma = err.
# --------------------------------------------------------------------------

_WEIGHTS = {'x': None, 'err': None}


def curve_fit(f, xdata, ydata, p0=None, **kw):
    x, err = _WEIGHTS['x'], _WEIGHTS['err']
    if x is not None and np.shape(xdata) == np.shape(x) and np.array_equal(xdata, x):
        kw.update(sigma=err, absolute_sigma=True)
    popt, pcov = _scipy_curve_fit(f, xdata, ydata, p0, **kw)
    b = kw.get('bounds')
    if b is not None and np.ndim(b[0]) == 1:   # parameters held fixed (fix_params): error 0
        lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
        fixed = (hi - lo) <= 1e-6*np.maximum(1.0, np.abs(popt))
        pcov = np.array(pcov, float)
        pcov[fixed, :] = 0.0; pcov[:, fixed] = 0.0
    return popt, pcov


class weighted:
    """with weighted(x, err): fits to (x, y) are weighted by 1/err².
    err None = unweighted (the notebook). Errors below 10% of the median error
    are raised to it, so one lucky point cannot pin the fit."""

    def __init__(self, x, err):
        self.x = None if err is None else np.asarray(x, float)
        self.err = None if err is None else clean_err(err)

    def __enter__(self):
        _WEIGHTS.update(x=self.x, err=self.err)
        return self

    def __exit__(self, *exc):
        _WEIGHTS.update(x=None, err=None)


def clean_err(err):
    e = np.asarray(err, float).copy()
    good = np.isfinite(e) & (e > 0)
    floor = 0.1*np.median(e[good]) if good.any() else 1.0
    e[~good] = np.max(e[good]) if good.any() else 1.0
    return np.maximum(e, floor)


def chi2_red(res, err, n_par):
    """Reduced χ² of a residual with errors err (≈ 1 when the error bars are right)."""
    res = np.asarray(res, float)
    dof = max(res.size - n_par, 1)
    return float(np.sum((res/clean_err(err))**2)/dof)


# Figures keep the notebook's figsize and font sizes; a low dpi keeps the
# windows a sensible size on screen.
FIG_DPI = 50


def _figure(figsize):
    return Figure(figsize=figsize, dpi=FIG_DPI)


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def R2(x):
    return round(x, 2)


def R4(x):
    return round(x, 4)


# --------------------------------------------------------------------------
# lorentzians
# --------------------------------------------------------------------------

def lorentzian(x, x0, a, gam, y0):
    return a*gam**2 / (gam**2+(x-x0)**2) + y0


def lorentzian_2(x, x01, a1, gam1, x02, a2, gam2, y0):
    return a1*gam1**2 / (gam1**2+(x-x01)**2) + a2*gam2**2 / (gam2**2+(x-x02)**2) + y0


def lorentzian_3(x, x01, a1, gam1, x02, a2, gam2, x03, a3, gam3, y0):
    return a1*gam1**2 / (gam1**2+(x-x01)**2) + a2*gam2**2 / (gam2**2+(x-x02)**2) + a3*gam3**2 / (gam3**2+(x-x03)**2) + y0


# --------------------------------------------------------------------------
# preprocessing, as done in the notebook cells before each fit
# --------------------------------------------------------------------------

def prepare_trace(time, signal, x_shift=0.0, base_start=0, base_stop=1, start=0, stop=None):
    """x = time - x_shift; y = NormalizeData(signal) - mean(y[base_start:base_stop]);
    then crop to [start:stop]."""
    x = np.asarray(time, float) - x_shift
    y = NormalizeData(np.asarray(signal, float))
    if base_stop > base_start:
        y = y - np.mean(y[base_start:base_stop])
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    return x[start:stop], y[start:stop]


# Time steps that differ by less than this (relative) count as one step.
STEP_TOLERANCE = 0.1


def time_steps(x):
    """Sections of constant time step: list of (step, first x, last x, n points)."""
    x = np.asarray(x, float)
    dt = np.diff(x)
    out = []
    for k, d in enumerate(dt):
        if out and abs(d - out[-1][0]) <= STEP_TOLERANCE*abs(out[-1][0]):
            st, a, _, n = out[-1]
            out[-1] = ((st*(n-1) + d)/n, a, x[k+1], n + 1)
        else:
            out.append((d, x[k], x[k+1], 2))
    return out


def is_uniform(x):
    dt = np.diff(np.asarray(x, float))
    return dt.size > 0 and np.all(np.abs(dt - np.median(dt)) <= STEP_TOLERANCE*abs(np.median(dt)))


def _residual_fft(x, res, cut):
    """Amplitude spectrum of the residual (from point `cut` on).

    Uniform time steps: the notebook FFT, unchanged. Non-uniform steps (e.g.
    0.2 ps around t0 and 2 ps later): the FFT would assume the first step
    everywhere and get the frequency axis wrong, so the Fourier integral is
    computed directly on the real time points (each point weighted by its own
    time step), on the grid df = 1/T up to the Nyquist frequency of the most
    common step. For uniform data both give the same numbers."""
    x_cut = x[cut:]
    res_cut = res[cut:]
    if is_uniform(x_cut):
        fft_res = fft(res_cut)
        N = len(fft_res)
        timestep = abs(x_cut[1]-x_cut[0])
        fft_freq = fftfreq(N, d=timestep)[:N//2]
        fft_res = 2/N*abs(fft_res[:N//2])
        return x_cut, res_cut, fft_freq, fft_res
    t = np.asarray(x_cut, float)
    dt = np.diff(t)
    w = np.empty_like(t)
    w[0], w[-1] = dt[0], dt[-1]
    w[1:-1] = (dt[:-1] + dt[1:])/2
    T = w.sum()
    f_nyq = 1/(2*np.median(np.abs(dt)))
    fft_freq = np.arange(0, f_nyq, 1/T)
    phase = np.exp(-2j*np.pi*np.outer(fft_freq, t))
    fft_res = 2/T*np.abs(phase @ (w*np.asarray(res_cut, float)))
    return x_cut, res_cut, fft_freq, fft_res


# --------------------------------------------------------------------------
# thermal decay
# --------------------------------------------------------------------------

THERMAL_NAMES = ['t0', 'σ', 'a_exp', 'τ', 'y0']
THERMAL_P0 = (2, 0.05, 0.7, 500, 0)
THERMAL_CONS = ((-2, 0, 0.1, 10, -0.01), (15, 2, 1, 5000, 0.01))


def thermal_model(t, t0, σ, a_exp, t_exp, y0):
    return abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) + y0))**2


# Thermal-fit filter (see thermal_quality); the GUI can change these.
THERMAL_FILTER = dict(max_band=1.0, min_r2=0.3, bound_tol=0.001)


def thermal_quality(x, y, th, cons, max_band=None, min_r2=None, bound_tol=None):
    """Decide whether the thermal decay fit is worth showing.

    Rejected when any of these holds:
    - t0, σ, a_exp or τ ended on a bound of the fit (within bound_tol of the
      allowed range): the fit was stopped by the limits, not by the data;
    - the τ ± err band of the notebook plot (fit with τ-err and τ+err) opens
      wider than max_band times the signal range anywhere on the trace, i.e.
      the decay is not determined by the data;
    - the curve does not follow the data: R² = 1 - SS_res/SS_tot below min_r2.
    A τ much longer than the trace is fine as long as the band stays narrow.
    Returns (ok, reasons, numbers)."""
    f = THERMAL_FILTER
    max_band = f['max_band'] if max_band is None else max_band
    min_r2 = f['min_r2'] if min_r2 is None else min_r2
    bound_tol = f['bound_tol'] if bound_tol is None else bound_tol
    popt, err = np.asarray(th['popt'], float), np.asarray(th['err'], float)
    lo, hi = (np.asarray(c, float) for c in cons)
    reasons = []
    for k, name in enumerate(THERMAL_NAMES[:4]):
        span = hi[k] - lo[k]
        if span > 0 and (popt[k] - lo[k] <= bound_tol*span or hi[k] - popt[k] <= bound_tol*span):
            reasons.append(f'{name} = {R2(popt[k])} is on its bound [{lo[k]:g}, {hi[k]:g}]')
    t0, σ, a_exp, t_exp, y0 = popt[:5]
    xi = np.linspace(np.min(x), np.max(x), 1000)
    with np.errstate(all='ignore'):
        fl = thermal_model(xi, t0, σ, a_exp, t_exp - err[3], y0)
        fu = thermal_model(xi, t0, σ, a_exp, t_exp + err[3], y0)
    yr = np.ptp(y) or 1.0
    band = np.nanmax(np.abs(fu - fl))/yr if np.all(np.isfinite(err)) else np.inf
    if not np.isfinite(band) or band > max_band:
        reasons.append(f'τ = {R2(t_exp)} ± {R2(err[3])} ps: error band {band*100:.0f}% of the signal (max {max_band*100:.0f}%)')
    res = np.asarray(y) - np.asarray(th['fit'])
    r2 = 1 - np.sum(res**2)/np.sum((np.asarray(y) - np.mean(y))**2)
    if r2 < min_r2:
        reasons.append(f'R² = {r2:.2f} (min {min_r2:.2f})')
    return not reasons, reasons, dict(band=band, r2=r2)


def trace_only_figure(x, y, note):
    """Notebook layout of the thermal figure without the fit (fit rejected)."""
    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax2.plot(x,y,'-r',lw=2)
    for ax in (ax1, ax2):
        ax.axhline(y=0,lw=.5,color='k'); ax.axvline(x=0,lw=.5,color='k')
        ax.set_xlabel('Δt [ps]',fontsize=20); ax.set_ylabel('TG signal [a.u.]',fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_xlim([-5,40])
    fig.suptitle(note, fontsize=16, color='darkred')
    return fig


def fit_exp(x, y, p0, cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,y0: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) + y0))**2,x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,y0 = popt[0],popt[1],popt[2],popt[3],popt[4]
    err = np.sqrt(np.diag(pcov))

    x_int = np.linspace(np.min(x), np.max(x), 1000);

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) + y0))**2
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0))**2

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3]))]], headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]'], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x_int,fit_int,'-k',lw=2)
    ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.1)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,40])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=2)
    ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.1)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax2.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=THERMAL_NAMES, table=table, figures=[fig],
                title='Thermal decay')


# --------------------------------------------------------------------------
# FFT of the residual + lorentzian fit (gives the frequency guesses)
# --------------------------------------------------------------------------

def do_fft_lorentz_1(x,y,fit,p0,cons,cut,fft_xlim):

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500); f = interp1d(fft_freq,fft_res)
    fft_res_interp = f(fft_freq_interp)

    popt, pcov = curve_fit(lambda t,x01,a1,gam1,y0: lorentzian(t,x01,a1,gam1,y0),fft_freq_interp,fft_res_interp,p0,bounds=cons,maxfev=5000)
    x01,a1,gam1,y0 = popt[0],popt[1],popt[2],popt[3]
    err = np.sqrt(np.diag(pcov))

    table = tabulate([[str(R2(x01*1000))+' ± '+str(R2(err[0]*1000)), str(R4(gam1)), str(R2(2*gam1*1000))]], headers=['ν1 [GHz]', ' γ1 ', 'FWHM'], tablefmt='orgtbl')
    print(table); print()

    value = min(fft_freq_interp, key=lambda x:abs(x-x01)); freq1 = np.where(fft_freq_interp == value)

    fit_lor_tot = lorentzian(fft_freq_interp,x01,a1,gam1,y0); print()
    lor1 = lorentzian(fft_freq_interp,x01,a1,gam1,0);

    fig = _figure((30, 8))
    ax3 = fig.add_subplot(121) ; ax4 = fig.add_subplot(122)

    ax3.plot(x,res,'--k',label='residual')
    ax3.plot(x_cut,res_cut,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.plot(fft_freq_interp*1000,fit_lor_tot*100,'-k',lw=2)
    ax4.plot(fft_freq_interp*1000,lor1*100,color='g'); ax4.fill_between(fft_freq_interp*1000,lor1*100,color='g',alpha=.4)
    ax4.plot(x01*1000,fit_lor_tot[freq1]*100,'og',ms=8,mfc='w'); ax4.text(1.15*x01*1000,fit_lor_tot[freq1][0]*100,str(R2(x01*1000))+' GHz')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    return dict(fft_freq=fft_freq, fft_res=fft_res, fit_lor_tot=fit_lor_tot, popt=popt, err=err,
                table=table, figures=[fig], title='FFT residual, 1 lorentzian')


def do_fft_lorentz_2(x,y,fit,p0,cons,cut,fft_xlim):

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500);

    popt, pcov = curve_fit(lambda t,x01,a1,gam1,x02,a2,gam2,y0: lorentzian_2(t,x01,a1,gam1,x02,a2,gam2,y0),fft_freq,fft_res,p0,bounds=cons,maxfev=5000)
    x01,a1,gam1,x02,a2,gam2,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6]
    err = np.sqrt(np.diag(pcov))

    table = tabulate([[str(R2(x01*1000))+' ± '+str(R2(err[0]*1000)), str(R4(gam1)), str(R2(x02*1000))+' ± '+str(R2(err[3]*1000)), str(R4(gam2))]], headers=['ν1 [GHz]', ' γ1 ', 'ν2 [GHz]', ' γ2 '], tablefmt='orgtbl')
    print(table); print()

    value = min(fft_freq_interp, key=lambda x:abs(x-x01)); freq1 = np.where(fft_freq_interp == value)
    value = min(fft_freq_interp, key=lambda x:abs(x-x02)); freq2 = np.where(fft_freq_interp == value)

    fit_lor_tot = lorentzian_2(fft_freq_interp,x01,a1,gam1,x02,a2,gam2,y0); print()
    lor1 = lorentzian(fft_freq_interp,x01,a1,gam1,0); lor2 = lorentzian(fft_freq_interp,x02,a2,gam2,0);

    fig = _figure((30, 8))
    ax3 = fig.add_subplot(121) ; ax4 = fig.add_subplot(122)

    ax3.plot(x,res,'--k',label='residual')
    ax3.plot(x_cut,res_cut,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.plot(fft_freq_interp*1000,fit_lor_tot*100,'-k',lw=2)
    ax4.plot(fft_freq_interp*1000,lor1*100,color='g'); ax4.fill_between(fft_freq_interp*1000,lor1*100,color='g',alpha=.4)
    ax4.plot(fft_freq_interp*1000,lor2*100,color='r'); ax4.fill_between(fft_freq_interp*1000,lor2*100,color='r',alpha=.4)
    ax4.plot(x01*1000,fit_lor_tot[freq1]*100,'og',ms=8,mfc='w'); ax4.text(1.15*x01*1000,fit_lor_tot[freq1][0]*100,str(R2(x01*1000))+' GHz')
    ax4.plot(x02*1000,fit_lor_tot[freq2]*100,'or',ms=8,mfc='w'); ax4.text(1.075*x02*1000,fit_lor_tot[freq2][0]*100,str(R2(x02*1000))+' GHz')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    return dict(fft_freq=fft_freq, fft_res=fft_res, fit_lor_tot=fit_lor_tot, popt=popt, err=err,
                table=table, figures=[fig], title='FFT residual, 2 lorentzians')


def do_fft_lorentz_3(x,y,fit,p0,cons,cut,fft_xlim):

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500); f = interp1d(fft_freq,fft_res)
    fft_res_interp = f(fft_freq_interp)

    popt, pcov = curve_fit(lambda t,x01,a1,gam1,x02,a2,gam2,x03,a3,gam3,y0: lorentzian_3(t,x01,a1,gam1,x02,a2,gam2,x03,a3,gam3,y0),fft_freq_interp,fft_res_interp,p0,bounds=cons,maxfev=5000)
    x01,a1,gam1,x02,a2,gam2,x03,a3,gam3,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9]
    err = np.sqrt(np.diag(pcov))

    table = tabulate([[str(R2(x01*1000))+' ± '+str(R2(err[0]*1000)), str(R4(gam1)), str(R2(2*gam1*1000)), str(R2(x02*1000))+' ± '+str(R2(err[3]*1000)), str(R4(gam2)), str(R2(2*gam2*1000)), str(R2(x03*1000))+' ± '+str(R2(err[6]*1000)), str(R4(gam3)), str(R2(2*gam3*1000))]], headers=['ν1 [GHz]', ' γ1 ', 'FWHM', 'ν2 [GHz]', ' γ2 ', 'FWHM', 'ν3 [GHz]', ' γ3 ', 'FWHM'], tablefmt='orgtbl')
    print(table); print()

    value = min(fft_freq_interp, key=lambda x:abs(x-x01)); freq1 = np.where(fft_freq_interp == value)
    value = min(fft_freq_interp, key=lambda x:abs(x-x02)); freq2 = np.where(fft_freq_interp == value)
    value = min(fft_freq_interp, key=lambda x:abs(x-x03)); freq3 = np.where(fft_freq_interp == value)

    fit_lor_tot = lorentzian_3(fft_freq_interp,x01,a1,gam1,x02,a2,gam2,x03,a3,gam3,y0); print()
    lor1 = lorentzian(fft_freq_interp,x01,a1,gam1,0); lor2 = lorentzian(fft_freq_interp,x02,a2,gam2,0); lor3 = lorentzian(fft_freq_interp,x03,a3,gam3,0)

    fig = _figure((30, 8))
    ax3 = fig.add_subplot(121) ; ax4 = fig.add_subplot(122)

    ax3.plot(x,res,'--k',label='residual')
    ax3.plot(x_cut,res_cut,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.plot(fft_freq_interp*1000,fit_lor_tot*100,'-k',lw=2)
    ax4.plot(fft_freq_interp*1000,lor1*100,color='g'); ax4.fill_between(fft_freq_interp*1000,lor1*100,color='g',alpha=.4)
    ax4.plot(fft_freq_interp*1000,lor2*100,color='r'); ax4.fill_between(fft_freq_interp*1000,lor2*100,color='r',alpha=.4)
    ax4.plot(fft_freq_interp*1000,lor3*100,color='b'); ax4.fill_between(fft_freq_interp*1000,lor3*100,color='b',alpha=.4)
    ax4.plot(x01*1000,fit_lor_tot[freq1]*100,'og',ms=8,mfc='w'); ax4.text(1.15*x01*1000,fit_lor_tot[freq1][0]*100,str(R2(x01*1000))+' GHz')
    ax4.plot(x02*1000,fit_lor_tot[freq2]*100,'or',ms=8,mfc='w'); ax4.text(1.075*x02*1000,fit_lor_tot[freq2][0]*100,str(R2(x02*1000))+' GHz')
    ax4.plot(x03*1000,fit_lor_tot[freq3]*100,'ob',ms=8,mfc='w'); ax4.text(1.05*x03*1000,fit_lor_tot[freq3][0]*100,str(R2(x03*1000))+' GHz')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    return dict(fft_freq=fft_freq, fft_res=fft_res, fit_lor_tot=fit_lor_tot, popt=popt, err=err,
                table=table, figures=[fig], title='FFT residual, 3 lorentzians')


DO_FFT_LORENTZ = {1: do_fft_lorentz_1, 2: do_fft_lorentz_2, 3: do_fft_lorentz_3}


# --------------------------------------------------------------------------
# global fits
# --------------------------------------------------------------------------

FREQ1_DUMP1_NAMES = ['t0', 'σ', 'a_exp', 'τ', 'a1', 'ν1', 'ϕ1', 'τ1', 'y0']


def freq1_dump1(x,y,fft_res_exp,fit_lor_tot,cut,fft_xlim,p0,cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,a1,ν1,ϕ1,t1,y0: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) - a1*np.sin(2*np.pi*ν1*(t-t0)+ϕ1)*np.exp(-1/t1*(t-t0)) + y0))**2,x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,a1,ν1,ϕ1,t1,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8]
    err = np.sqrt(np.diag(pcov))

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) - a1*np.sin(2*np.pi*ν1*(x-t0)+ϕ1)*np.exp(-1/t1*(x-t0)) + y0))**2

    x_int = np.linspace(np.min(x), np.max(x), 1000);
    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) - a1*np.sin(2*np.pi*ν1*(x_int-t0)+ϕ1)*np.exp(-1/t1*(x_int-t0)) + y0))**2

    fit_thermal = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0))**2
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500);

    fit_lor_tot_rec = lorentzian(fft_freq_interp,ν1,a1,.5*err[5],0)

    least_square = np.sum(res**2)

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3])), \
                   str(R2(a1))+' ± '+str(R2(err[4])), str(R2(ν1*1000))+' ± '+str(R2(err[5]*1000)), str(R2(ϕ1))+' ± '+str(R2(err[6])), \
                   str(R2(t1))+' ± '+str(R2(err[7])), str(R2(y0))+' ± '+str(R2(err[8])), str(R4(least_square))]],
                   headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]', 'a1', 'ν1 [GHz]', 'ϕ1', 'τ1 [ps]', 'y0', 'least square'], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x,fit,'-k',lw=2)
    ax1.plot(x_int,fit_int,'-k',lw=.5)
    ax1.plot(x_int,fit_thermal,'-b',lw=.5); ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,50])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2)
    ax2.plot(x,fit,'-k',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=.5)
    ax2.plot(x_int,fit_thermal,'-b',lw=.5); ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig2 = _figure((30, 8))
    ax3 = fig2.add_subplot(131); ax4 = fig2.add_subplot(132); ax5 = fig2.add_subplot(133)

    ax3.plot(x,res*100,'--k',label='residual')
    ax3.plot(x_cut,res_cut*100,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual [%]',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    ax5.plot(fft_freq*1000, NormalizeData(fft_res_exp), '--b',linewidth=1.5, label='FFT residual')
    ax5.plot(np.linspace(np.min(fft_freq), np.max(fft_freq), len(fit_lor_tot))*1000,NormalizeData(fit_lor_tot),'-r',lw=2)
    ax5.plot(fft_freq_interp*1000,NormalizeData(fit_lor_tot_rec),'-k',lw=2)
    ax5.set_xlim([-1,fft_xlim])
    ax5.set_ylabel('FFT amplitude [a.u.]',fontsize=20); ax5.set_xlabel('ν [GHz]',fontsize=20); ax5.legend(fontsize=14)
    ax5.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=FREQ1_DUMP1_NAMES, table=table,
                figures=[fig, fig2], title='1 freq 1 dump',
                x_int=x_int, fit_int=fit_int, fit_thermal=fit_thermal, res=res,
                fft_freq=fft_freq, fft_res=fft_res, least_square=least_square)


FREQ2_DUMP1_NAMES = ['t0', 'σ', 'a_exp', 'τ', 'a1', 'ν1', 'ϕ1', 'a2', 'ν2', 'ϕ2', 'τ2', 'y0']


def freq2_dump1(x,y,fft_res_exp,fit_lor_tot,cut,fft_xlim,plot_Δt,p0,cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,y0: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) - a1*np.sin(2*np.pi*ν1*(t-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(t-t0)+ϕ2)*np.exp(-1/t2*(t-t0)) + y0))**2,x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9],popt[10],popt[11]
    err = np.sqrt(np.diag(pcov))

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) - a1*np.sin(2*np.pi*ν1*(x-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x-t0)+ϕ2)*np.exp(-1/t2*(x-t0)) + y0))**2

    x_int = np.linspace(np.min(x), np.max(x), 1000);
    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) - a1*np.sin(2*np.pi*ν1*(x_int-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x_int-t0)+ϕ2)*np.exp(-1/t2*(x_int-t0)) + y0))**2

    fit_thermal = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0))**2
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500);

    fit_lor_tot_rec = lorentzian(fft_freq_interp,ν1,a1,.5*err[5],0) + lorentzian(fft_freq_interp,ν2,a2,.5*err[8],0)

    least_square = np.sum(res**2)

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3])), \
                   str(R2(a1))+' ± '+str(R2(err[4])), str(R2(ν1*1000))+' ± '+str(R2(err[5]*1000)), str(R2(ϕ1))+' ± '+str(R2(err[6])), str(R2(a2))+' ± '+str(R2(err[7])), str(R2(ν2*1000))+' ± '+str(R2(err[8]*1000)), str(R2(ϕ2))+' ± '+str(R2(err[9])), \
                   str(R2(t2))+' ± '+str(R2(err[10])), str(R2(y0))+' ± '+str(R2(err[11])), str(R4(least_square))]],
                   headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]', 'a1', 'ν1 [GHz]', 'ϕ1', 'a2', 'ν2 [GHz]', 'ϕ2', 'τ2 [ps]', 'y0', 'least square'], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x,fit,'-k',lw=2)
    ax1.plot(x_int,fit_int,'-k',lw=.5)
    ax1.plot(x_int,fit_thermal,'-b',lw=.5);
    if plot_Δt == 1:
        ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,50])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2,label='TG signal')
    # ax2.plot(x,fit,'-k',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=1,label='Fit')
    ax2.plot(x_int,fit_thermal,'-b',lw=1,label='Thermal decay');
    if plot_Δt == 1:
        ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20); ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig2 = _figure((30, 8))
    ax3 = fig2.add_subplot(131); ax4 = fig2.add_subplot(132); ax5 = fig2.add_subplot(133)

    ax3.plot(x,res*100,'--k',label='residual')
    ax3.plot(x_cut,res_cut*100,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual [%]',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    ax5.plot(fft_freq*1000, NormalizeData(fft_res_exp), '--b',linewidth=1.5, label='FFT residual')
    ax5.plot(np.linspace(np.min(fft_freq), np.max(fft_freq), len(fit_lor_tot))*1000,NormalizeData(fit_lor_tot),'-r',lw=2)
    ax5.plot(fft_freq_interp*1000,NormalizeData(fit_lor_tot_rec),'-k',lw=2)
    ax5.axvline(x=ν1*1000,color='k',lw=1,ls='--'); ax5.axvline(x=ν2*1000,color='k',lw=1,ls='--')
    ax5.set_xlim([-1,fft_xlim])
    ax5.set_ylabel('FFT amplitude [a.u.]',fontsize=20); ax5.set_xlabel('ν [GHz]',fontsize=20); ax5.legend(fontsize=14)
    ax5.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=FREQ2_DUMP1_NAMES, table=table,
                figures=[fig, fig2], title='2 freq 1 dump',
                x_int=x_int, fit_int=fit_int, fit_thermal=fit_thermal, res=res,
                fft_freq=fft_freq, fft_res=fft_res, least_square=least_square)


FREQ3_DUMP2_NAMES = ['t0', 'σ', 'a_exp', 'τ', 'a1', 'ν1', 'ϕ1', 'a2', 'ν2', 'ϕ2', 'τ2', 'a3', 'ν3', 'ϕ3', 'τ3', 'y0']


def freq3_dump2(x,y,fft_res_exp,fit_lor_tot,cut,fft_xlim,plot_Δt,p0,cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,a3,ν3,ϕ3,t3,y0: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) - a1*np.sin(2*np.pi*ν1*(t-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(t-t0)+ϕ2)*np.exp(-1/t2*(t-t0)) - a3*np.sin(2*np.pi*ν3*(t-t0)+ϕ3)*np.exp(-1/t3*(t-t0)) + y0))**2,x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,a3,ν3,ϕ3,t3,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9],popt[10],popt[11],popt[12],popt[13],popt[14],popt[15]
    err = np.sqrt(np.diag(pcov))

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) - a1*np.sin(2*np.pi*ν1*(x-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x-t0)+ϕ2)*np.exp(-1/t2*(x-t0)) - a3*np.sin(2*np.pi*ν3*(x-t0)+ϕ3)*np.exp(-1/t3*(x-t0)) + y0))**2

    x_int = np.linspace(np.min(x), np.max(x), 1000);
    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) - a1*np.sin(2*np.pi*ν1*(x_int-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x_int-t0)+ϕ2)*np.exp(-1/t2*(x_int-t0)) - a3*np.sin(2*np.pi*ν3*(x_int-t0)+ϕ3)*np.exp(-1/t3*(x_int-t0)) + y0))**2

    fit_thermal = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0))**2
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500);

    fit_lor_tot_rec = lorentzian(fft_freq_interp,ν1,a1,.5*err[5],0) + lorentzian(fft_freq_interp,ν2,a2,.5*err[8],0) + lorentzian(fft_freq_interp,ν3,a3,.5*err[12],0)

    least_square = np.sum(res**2)

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3])), \
                   str(R2(a1))+' ± '+str(R2(err[4])), str(R2(ν1*1000))+' ± '+str(R2(err[5]*1000)), str(R2(ϕ1))+' ± '+str(R2(err[6])), str(R2(a2))+' ± '+str(R2(err[7])), str(R2(ν2*1000))+' ± '+str(R2(err[8]*1000)), str(R2(ϕ2))+' ± '+str(R2(err[9])), \
                   str(R2(t2))+' ± '+str(R2(err[10])), str(R2(a3))+' ± '+str(R2(err[11])), str(R2(ν3*1000))+' ± '+str(R2(err[12]*1000)), str(R2(ϕ3))+' ± '+str(R2(err[13])), str(R2(t3))+' ± '+str(R2(err[14])), str(R2(y0))+' ± '+str(R2(err[15])), str(R4(least_square))]],
                   headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]', 'a1', 'ν1 [GHz]', 'ϕ1', 'a2', 'ν2 [GHz]', 'ϕ2', 'τ2 [ps]', 'a3', 'ν3 [GHz]', 'ϕ3', 'τ3 [ps]', 'y0', 'least square'], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x,fit,'-k',lw=2)
    ax1.plot(x_int,fit_int,'-k',lw=.5)
    ax1.plot(x_int,fit_thermal,'-b',lw=.5);
    if plot_Δt == 1:
        ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,50])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2,label='TG signal')
    # ax2.plot(x,fit,'-k',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=1,label='Fit')
    ax2.plot(x_int,fit_thermal,'-b',lw=1,label='Thermal decay');
    if plot_Δt == 1:
        ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20); ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig2 = _figure((30, 8))
    ax3 = fig2.add_subplot(131); ax4 = fig2.add_subplot(132); ax5 = fig2.add_subplot(133)

    ax3.plot(x,res*100,'--k',label='residual')
    ax3.plot(x_cut,res_cut*100,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual [%]',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    ax5.plot(fft_freq*1000, NormalizeData(fft_res_exp), '--b',linewidth=1.5, label='FFT residual')
    ax5.plot(np.linspace(np.min(fft_freq), np.max(fft_freq), len(fit_lor_tot))*1000,NormalizeData(fit_lor_tot),'-r',lw=2)
    ax5.plot(fft_freq_interp*1000,NormalizeData(fit_lor_tot_rec),'-k',lw=2)
    ax5.set_xlim([-1,fft_xlim])
    ax5.set_ylabel('FFT amplitude [a.u.]',fontsize=20); ax5.set_xlabel('ν [GHz]',fontsize=20); ax5.legend(fontsize=14)
    ax5.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=FREQ3_DUMP2_NAMES, table=table,
                figures=[fig, fig2], title='3 freq 2 dump',
                x_int=x_int, fit_int=fit_int, fit_thermal=fit_thermal, res=res,
                fft_freq=fft_freq, fft_res=fft_res, least_square=least_square)


# --------------------------------------------------------------------------
# guesses: FFT peaks -> lorentzian p0/cons -> global-fit p0/cons
# --------------------------------------------------------------------------

def residual_fft(x, y, fit, cut):
    """FFT of the thermal-fit residual, exactly as the functions above compute it."""
    _, _, fft_freq, fft_res = _residual_fft(x, y - fit, cut)
    return fft_freq, fft_res


def find_fft_peaks(fft_freq, fft_res, n, nu_guess_ghz=None, nu_min_ghz=0.0, nu_max_ghz=None):
    """Frequencies (THz) of the n strongest FFT peaks between nu_min_ghz and
    nu_max_ghz (None = no upper limit; the GUI passes the FFT plot limit, so
    a finely sampled stretch near t0 cannot pull in THz noise peaks).

    nu_guess_ghz: optional list of user guesses in GHz; entries that are
    None/NaN are filled from the detected peaks.
    """
    keep = fft_freq*1000 > max(nu_min_ghz, 0.0)
    if nu_max_ghz:
        keep &= fft_freq*1000 <= nu_max_ghz
    keep[0] = False
    idx, _ = find_peaks(np.where(keep, fft_res, 0.0))
    idx = idx[np.argsort(fft_res[idx])[::-1]]
    auto = list(fft_freq[idx])
    out = []
    guesses = list(nu_guess_ghz or []) + [None]*n
    for g in guesses[:n]:
        if g is not None and np.isfinite(g):
            out.append(g/1000)
        else:
            while auto and any(abs(auto[0]-o) < 1e-12 for o in out):
                auto.pop(0)
            out.append(auto.pop(0) if auto else fft_freq[min(len(fft_freq)-1, 1+len(out))])
    return out


def freq_windows(n, spans_ghz=None, df=0.001):
    """Half-width (THz) of the allowed range of each of n frequencies.

    spans_ghz: user spans ±Δν in GHz, one per frequency (a single value is
    used for all; missing/blank entries get the default max(3 GHz, 2 bins)),
    as ν_l = ν - Δν, ν_u = ν + Δν in the notebook.
    """
    default = max(0.003, 2*df)
    spans = [v for v in (spans_ghz or [])]
    if len(spans) == 1:
        spans = spans*n
    out = []
    for k in range(n):
        v = spans[k] if k < len(spans) else None
        out.append(v/1000 if v is not None and np.isfinite(v) and v > 0 else default)
    return out


def guess_lorentz(fft_freq, fft_res, nus, spans_ghz=None):
    """p0/cons for do_fft_lorentz_N from peak frequencies (THz); each ν may
    move by ±spans_ghz (see freq_windows)."""
    df = fft_freq[1]-fft_freq[0] if len(fft_freq) > 1 else 0.001
    wins = freq_windows(len(nus), spans_ghz, df)
    amax = float(np.max(fft_res)) if len(fft_res) else 1.0
    p0, lo, hi = [], [], []
    for nu, win in zip(nus, wins):
        a = float(np.interp(nu, fft_freq, fft_res))
        p0 += [nu, max(a, 1e-6), max(df, 0.002)]
        lo += [max(nu-win, 0.0), 0.0, 1e-4]
        hi += [nu+win, 3*amax, 0.05]
    p0.append(0.0); lo.append(0.0); hi.append(amax)
    return _clip(p0, lo, hi), (tuple(lo), tuple(hi))


# --------------------------------------------------------------------------
# electronic peak / coherent spike: the same four models plus a fast term,
# in one of two forms (set_el_mode; the GUI has a switch):
#
# 'cs' coherent spike (default; for resonant pumping, circular pumps and a
#      linear probe): I = |I_TG ± B·G|² + |C·G|², G = exp(-((t-t_cs)/σ_cs)²).
#      B·G interferes with the TG field (the sign of B picks + or -); C·G is
#      the orthogonal polarisation, with its own amplitude, and only adds
#      (C ≥ 0: only C² enters). G has its own t_cs, σ_cs: the TG erf (t0, σ)
#      comes from the thermal fit, where σ can be the slow build-up of the
#      grating (magnetite: 0.6 ps) rather than the time resolution.
# 'free' free electronic peak: |I_TG + a_el·el_peak(t; t_el, σ_el, τ_el)|²,
#      a step at its own start t_el (rise σ_el) decaying with τ_el, for when
#      the slow part of the signal starts later and rises more slowly than
#      the spike (magnetite Scan_035: spike at 0.07 ps, σ ≈ 0.5 ps).
#
# The fit functions are the notebook ones with only these changes: the extra
# parameters at the end (*el), el_in inside the modulus and el_out after it,
# a finer x_int around the spike, the spike drawn in green, extra columns.
# --------------------------------------------------------------------------

def el_peak(t, t_el, σ_el, τ_el):
    """exp(-(t-t_el)/τ_el) for t > t_el, convolved with exp(-(t/σ_el)²) and
    normalised to the step height (the peak is at most 1). erfcx keeps it
    finite for any σ_el/τ_el."""
    u = np.asarray(t, dtype=float) - t_el
    z = σ_el/(2*τ_el) - u/σ_el
    with np.errstate(over='ignore', invalid='ignore'):
        neg = 0.5*np.exp(σ_el**2/(4*τ_el**2) - u/τ_el)*erfc(z)
        pos = 0.5*np.exp(-(u/σ_el)**2)*erfcx(z)
    return np.where(z < 0, neg, pos)


def _cs(t, t0, σ, B):
    return B*np.exp(-((np.asarray(t, dtype=float)-t0)/σ)**2)


EL_MODES = {
    'cs': dict(names=['B', 'C', 't_cs', 'σ_cs'], p0=(0.3, 0.1, 0.0, 0.05),
               cons=((-3.0, 0.0, -2.0, 0.005), (3.0, 3.0, 15.0, 0.5)), title='coherent spike',
               headers=['B', 'C', 't_cs [ps]', 'σ_cs [fs]'], label='Coherent spike |B·G|² + |C·G|²',
               inside=lambda t, t0, σ, el: _cs(t, el[2], el[3], el[0]),
               outside=lambda t, t0, σ, el: _cs(t, el[2], el[3], el[1])**2,
               centre=lambda t0, el: el[2] - 0.5,
               cells=lambda el, err: [f'{R2(el[0])} ± {R2(err[0])}', f'{R2(el[1])} ± {R2(err[1])}',
                                      f'{R2(el[2])} ± {R2(err[2])}', f'{R2(el[3]*1000)} ± {R2(err[3]*1000)}']),
    'free': dict(names=['a_el', 't_el', 'σ_el', 'τ_el'], p0=(0.5, 0.0, 0.03, 0.1),
                 cons=((-3.0, -2.0, 0.005, 0.005), (3.0, 15.0, 0.2, 2.0)), title='electronic peak',
                 headers=['a_el', 't_el [ps]', 'σ_el [fs]', 'τ_el [fs]'], label='Electronic peak',
                 inside=lambda t, t0, σ, el: el[0]*el_peak(t, *el[1:4]),
                 outside=lambda t, t0, σ, el: 0.0,
                 centre=lambda t0, el: el[1],
                 cells=lambda el, err: [f'{R2(el[0])} ± {R2(err[0])}', f'{R2(el[1])} ± {R2(err[1])}',
                                        f'{R2(el[2]*1000)} ± {R2(err[2]*1000)}',
                                        f'{R2(el[3]*1000)} ± {R2(err[3]*1000)}']),
}
EL_MODE = 'cs'


def el_in(t, t0, σ, el):  return EL_MODES[EL_MODE]['inside'](t, t0, σ, el)
def el_out(t, t0, σ, el): return EL_MODES[EL_MODE]['outside'](t, t0, σ, el)
def el_centre(t0, el):    return EL_MODES[EL_MODE]['centre'](t0, el)
def el_cells(el, err):    return EL_MODES[EL_MODE]['cells'](el, err)
def el_headers():         return EL_MODES[EL_MODE]['headers']
def el_title():           return EL_MODES[EL_MODE]['title']
def el_label():           return EL_MODES[EL_MODE]['label']


def el_curve(t, t0, σ, el):
    """The fast term on its own (TG field set to zero), drawn in green."""
    return np.abs(el_in(t, t0, σ, el))**2 + el_out(t, t0, σ, el)


def guess_el(x, y):
    """Start values for the current mode from the data. The spike is taken
    at the steepest rise of the 3-point smoothed trace, its height from the
    maximum in the 0.3 ps after it (y is normalised to max 1, the field is √y).
    'free': a_el = √height, t_el there, σ_el 30 fs, τ_el 100 fs.
    'cs': B = C = √(height/2) (the two spike terms add up to B² + C² at the
    spike), t_cs at the spike maximum, σ_cs 50 fs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ys = np.convolve(y, np.ones(3)/3, mode='same')
    dydx = np.diff(ys)/np.maximum(np.diff(x), 1e-9)
    i = int(np.argmax(dydx[1:-1])) + 1 if len(dydx) > 2 else 0
    t_el = x[i]
    after = (x >= t_el) & (x <= t_el + 0.3)
    peak = max(float(np.max(y[after]) if after.any() else np.max(y)), 1e-6)
    t_max = float(x[after][np.argmax(y[after])]) if after.any() else t_el
    v = (np.sqrt(peak/2), np.sqrt(peak/2), t_max, 0.05) if EL_MODE == 'cs' else (np.sqrt(peak), t_el, 0.03, 0.1)
    lo, hi = EL_CONS
    return tuple(float(np.clip(a, l, h)) for a, l, h in zip(v, lo, hi))


def spike_window(x, y, max_len=1.0):
    """Time window (ps) of the fast spike, for the thermal fit to leave out:
    from 0.1 ps before the steepest rise to the first minimum of the
    3-point smoothed trace after the spike maximum (at most max_len ps
    after the rise)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ys = np.convolve(y, np.ones(3)/3, mode='same')
    dydx = np.diff(ys)/np.maximum(np.diff(x), 1e-9)
    i = int(np.argmax(dydx[1:-1])) + 1 if len(dydx) > 2 else 0
    t_rise = x[i]
    after = np.where((x >= t_rise) & (x <= t_rise + max_len))[0]
    if len(after) < 3:
        return t_rise - 0.1, t_rise + 0.1
    k = after[int(np.argmax(ys[after]))]
    end = after[-1]
    for j in range(k + 1, after[-1]):
        if ys[j] <= ys[j - 1] and ys[j] < ys[j + 1]:
            end = j
            break
    return float(t_rise - 0.1), float(x[end])


def fix_params(p0, cons, names, values):
    """p0/cons with the parameters in values ({name: value}) held at that
    value (bounds ± a hair, since curve_fit needs lower < upper)."""
    p0, lo, hi = list(p0), list(cons[0]), list(cons[1])
    for k, n in enumerate(names):
        if n in values:
            v = float(values[n]); eps = 1e-7*max(1.0, abs(v))
            p0[k], lo[k], hi[k] = v, v - eps, v + eps
    return p0, (tuple(lo), tuple(hi))


def set_el_mode(mode):
    """Switch the fast term ('cs' or 'free') for every _el fit."""
    global EL_MODE, EL_NAMES, EL_P0, EL_CONS
    m = EL_MODES[mode]
    EL_MODE, EL_NAMES, EL_P0, EL_CONS = mode, list(m['names']), m['p0'], m['cons']
    if 'GLOBAL_MODELS_EL' in globals():
        GLOBAL_MODELS_EL.update({1: (freq1_dump1_el, FREQ1_DUMP1_NAMES + EL_NAMES),
                                 2: (freq2_dump1_el, FREQ2_DUMP1_NAMES + EL_NAMES),
                                 3: (freq3_dump2_el, FREQ3_DUMP2_NAMES + EL_NAMES)})


EL_NAMES = list(EL_MODES[EL_MODE]['names'])
EL_P0 = EL_MODES[EL_MODE]['p0']
EL_CONS = EL_MODES[EL_MODE]['cons']


def fit_exp_el(x, y, p0, cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,y0,*el: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) + y0) + el_in(t,t0,σ,el))**2 + el_out(t,t0,σ,el),x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,y0 = popt[0],popt[1],popt[2],popt[3],popt[4]
    el = popt[5:]
    err = np.sqrt(np.diag(pcov))

    x_int = np.sort(np.r_[np.linspace(np.min(x), np.max(x), 1000), np.clip(el_centre(t0,el) + np.linspace(-0.5, 1.5, 401), np.min(x), np.max(x))]);

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) + y0) + el_in(x,t0,σ,el))**2 + el_out(x,t0,σ,el)
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0) + el_in(x_int,t0,σ,el))**2 + el_out(x_int,t0,σ,el)

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3])), *el_cells(el, err[5:])]], headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]', *el_headers()], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x_int,fit_int,'-k',lw=2)
    ax1.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1.5)
    ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.1)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,40])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=2)
    ax2.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1,label=el_label())
    ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.1)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax2.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=THERMAL_NAMES + EL_NAMES, table=table, figures=[fig],
                title='Thermal decay + ' + el_title())


def freq1_dump1_el(x,y,fft_res_exp,fit_lor_tot,cut,fft_xlim,p0,cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,a1,ν1,ϕ1,t1,y0,*el: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) - a1*np.sin(2*np.pi*ν1*(t-t0)+ϕ1)*np.exp(-1/t1*(t-t0)) + y0) + el_in(t,t0,σ,el))**2 + el_out(t,t0,σ,el),x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,a1,ν1,ϕ1,t1,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8]
    el = popt[9:]
    err = np.sqrt(np.diag(pcov))

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) - a1*np.sin(2*np.pi*ν1*(x-t0)+ϕ1)*np.exp(-1/t1*(x-t0)) + y0) + el_in(x,t0,σ,el))**2 + el_out(x,t0,σ,el)

    x_int = np.sort(np.r_[np.linspace(np.min(x), np.max(x), 1000), np.clip(el_centre(t0,el) + np.linspace(-0.5, 1.5, 401), np.min(x), np.max(x))]);
    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) - a1*np.sin(2*np.pi*ν1*(x_int-t0)+ϕ1)*np.exp(-1/t1*(x_int-t0)) + y0) + el_in(x_int,t0,σ,el))**2 + el_out(x_int,t0,σ,el)

    fit_thermal = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0))**2
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500);

    fit_lor_tot_rec = lorentzian(fft_freq_interp,ν1,a1,.5*err[5],0)

    least_square = np.sum(res**2)

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3])), \
                   str(R2(a1))+' ± '+str(R2(err[4])), str(R2(ν1*1000))+' ± '+str(R2(err[5]*1000)), str(R2(ϕ1))+' ± '+str(R2(err[6])), \
                   str(R2(t1))+' ± '+str(R2(err[7])), str(R2(y0))+' ± '+str(R2(err[8])), *el_cells(el, err[9:]), str(R4(least_square))]],
                   headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]', 'a1', 'ν1 [GHz]', 'ϕ1', 'τ1 [ps]', 'y0', *el_headers(), 'least square'], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x,fit,'-k',lw=2)
    ax1.plot(x_int,fit_int,'-k',lw=.5)
    ax1.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1.5)
    ax1.plot(x_int,fit_thermal,'-b',lw=.5); ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,50])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2)
    ax2.plot(x,fit,'-k',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=.5)
    ax2.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1,label=el_label())
    ax2.plot(x_int,fit_thermal,'-b',lw=.5); ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig2 = _figure((30, 8))
    ax3 = fig2.add_subplot(131); ax4 = fig2.add_subplot(132); ax5 = fig2.add_subplot(133)

    ax3.plot(x,res*100,'--k',label='residual')
    ax3.plot(x_cut,res_cut*100,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual [%]',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    ax5.plot(fft_freq*1000, NormalizeData(fft_res_exp), '--b',linewidth=1.5, label='FFT residual')
    ax5.plot(np.linspace(np.min(fft_freq), np.max(fft_freq), len(fit_lor_tot))*1000,NormalizeData(fit_lor_tot),'-r',lw=2)
    ax5.plot(fft_freq_interp*1000,NormalizeData(fit_lor_tot_rec),'-k',lw=2)
    ax5.set_xlim([-1,fft_xlim])
    ax5.set_ylabel('FFT amplitude [a.u.]',fontsize=20); ax5.set_xlabel('ν [GHz]',fontsize=20); ax5.legend(fontsize=14)
    ax5.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=FREQ1_DUMP1_NAMES + EL_NAMES, table=table,
                figures=[fig, fig2], title='1 freq 1 dump + ' + el_title(),
                x_int=x_int, fit_int=fit_int, fit_thermal=fit_thermal, res=res,
                fft_freq=fft_freq, fft_res=fft_res, least_square=least_square)


def freq2_dump1_el(x,y,fft_res_exp,fit_lor_tot,cut,fft_xlim,plot_Δt,p0,cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,y0,*el: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) - a1*np.sin(2*np.pi*ν1*(t-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(t-t0)+ϕ2)*np.exp(-1/t2*(t-t0)) + y0) + el_in(t,t0,σ,el))**2 + el_out(t,t0,σ,el),x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9],popt[10],popt[11]
    el = popt[12:]
    err = np.sqrt(np.diag(pcov))

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) - a1*np.sin(2*np.pi*ν1*(x-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x-t0)+ϕ2)*np.exp(-1/t2*(x-t0)) + y0) + el_in(x,t0,σ,el))**2 + el_out(x,t0,σ,el)

    x_int = np.sort(np.r_[np.linspace(np.min(x), np.max(x), 1000), np.clip(el_centre(t0,el) + np.linspace(-0.5, 1.5, 401), np.min(x), np.max(x))]);
    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) - a1*np.sin(2*np.pi*ν1*(x_int-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x_int-t0)+ϕ2)*np.exp(-1/t2*(x_int-t0)) + y0) + el_in(x_int,t0,σ,el))**2 + el_out(x_int,t0,σ,el)

    fit_thermal = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0))**2
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500);

    fit_lor_tot_rec = lorentzian(fft_freq_interp,ν1,a1,.5*err[5],0) + lorentzian(fft_freq_interp,ν2,a2,.5*err[8],0)

    least_square = np.sum(res**2)

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3])), \
                   str(R2(a1))+' ± '+str(R2(err[4])), str(R2(ν1*1000))+' ± '+str(R2(err[5]*1000)), str(R2(ϕ1))+' ± '+str(R2(err[6])), str(R2(a2))+' ± '+str(R2(err[7])), str(R2(ν2*1000))+' ± '+str(R2(err[8]*1000)), str(R2(ϕ2))+' ± '+str(R2(err[9])), \
                   str(R2(t2))+' ± '+str(R2(err[10])), str(R2(y0))+' ± '+str(R2(err[11])), *el_cells(el, err[12:]), str(R4(least_square))]],
                   headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]', 'a1', 'ν1 [GHz]', 'ϕ1', 'a2', 'ν2 [GHz]', 'ϕ2', 'τ2 [ps]', 'y0', *el_headers(), 'least square'], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x,fit,'-k',lw=2)
    ax1.plot(x_int,fit_int,'-k',lw=.5)
    ax1.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1.5)
    ax1.plot(x_int,fit_thermal,'-b',lw=.5);
    if plot_Δt == 1:
        ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,50])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2,label='TG signal')
    # ax2.plot(x,fit,'-k',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=1,label='Fit')
    ax2.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1,label=el_label())
    ax2.plot(x_int,fit_thermal,'-b',lw=1,label='Thermal decay');
    if plot_Δt == 1:
        ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20); ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig2 = _figure((30, 8))
    ax3 = fig2.add_subplot(131); ax4 = fig2.add_subplot(132); ax5 = fig2.add_subplot(133)

    ax3.plot(x,res*100,'--k',label='residual')
    ax3.plot(x_cut,res_cut*100,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual [%]',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    ax5.plot(fft_freq*1000, NormalizeData(fft_res_exp), '--b',linewidth=1.5, label='FFT residual')
    ax5.plot(np.linspace(np.min(fft_freq), np.max(fft_freq), len(fit_lor_tot))*1000,NormalizeData(fit_lor_tot),'-r',lw=2)
    ax5.plot(fft_freq_interp*1000,NormalizeData(fit_lor_tot_rec),'-k',lw=2)
    ax5.axvline(x=ν1*1000,color='k',lw=1,ls='--'); ax5.axvline(x=ν2*1000,color='k',lw=1,ls='--')
    ax5.set_xlim([-1,fft_xlim])
    ax5.set_ylabel('FFT amplitude [a.u.]',fontsize=20); ax5.set_xlabel('ν [GHz]',fontsize=20); ax5.legend(fontsize=14)
    ax5.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=FREQ2_DUMP1_NAMES + EL_NAMES, table=table,
                figures=[fig, fig2], title='2 freq 1 dump + ' + el_title(),
                x_int=x_int, fit_int=fit_int, fit_thermal=fit_thermal, res=res,
                fft_freq=fft_freq, fft_res=fft_res, least_square=least_square)


def freq3_dump2_el(x,y,fft_res_exp,fit_lor_tot,cut,fft_xlim,plot_Δt,p0,cons):

    popt, pcov = curve_fit(lambda t,t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,a3,ν3,ϕ3,t3,y0,*el: abs(0.5*(1+erf((t-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(t-t0)) - a1*np.sin(2*np.pi*ν1*(t-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(t-t0)+ϕ2)*np.exp(-1/t2*(t-t0)) - a3*np.sin(2*np.pi*ν3*(t-t0)+ϕ3)*np.exp(-1/t3*(t-t0)) + y0) + el_in(t,t0,σ,el))**2 + el_out(t,t0,σ,el),x,y,p0,bounds=cons,maxfev=5000)
    t0,σ,a_exp,t_exp,a1,ν1,ϕ1,a2,ν2,ϕ2,t2,a3,ν3,ϕ3,t3,y0 = popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9],popt[10],popt[11],popt[12],popt[13],popt[14],popt[15]
    el = popt[16:]
    err = np.sqrt(np.diag(pcov))

    fit = abs(0.5*(1+erf((x-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x-t0)) - a1*np.sin(2*np.pi*ν1*(x-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x-t0)+ϕ2)*np.exp(-1/t2*(x-t0)) - a3*np.sin(2*np.pi*ν3*(x-t0)+ϕ3)*np.exp(-1/t3*(x-t0)) + y0) + el_in(x,t0,σ,el))**2 + el_out(x,t0,σ,el)

    x_int = np.sort(np.r_[np.linspace(np.min(x), np.max(x), 1000), np.clip(el_centre(t0,el) + np.linspace(-0.5, 1.5, 401), np.min(x), np.max(x))]);
    fit_int = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) - a1*np.sin(2*np.pi*ν1*(x_int-t0)+ϕ1) - a2*np.sin(2*np.pi*ν2*(x_int-t0)+ϕ2)*np.exp(-1/t2*(x_int-t0)) - a3*np.sin(2*np.pi*ν3*(x_int-t0)+ϕ3)*np.exp(-1/t3*(x_int-t0)) + y0) + el_in(x_int,t0,σ,el))**2 + el_out(x_int,t0,σ,el)

    fit_thermal = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/t_exp*(x_int-t0)) + y0))**2
    fit_l = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp-err[3])*(x_int-t0)) + y0))**2
    fit_u = abs(0.5*(1+erf((x_int-t0)/σ)) * (a_exp*np.exp(-1/(t_exp+err[3])*(x_int-t0)) + y0))**2

    res = y - fit;    x_cut, res_cut, fft_freq, fft_res = _residual_fft(x, res, cut)

    fft_freq_interp = np.linspace(np.min(fft_freq), np.max(fft_freq), 500);

    fit_lor_tot_rec = lorentzian(fft_freq_interp,ν1,a1,.5*err[5],0) + lorentzian(fft_freq_interp,ν2,a2,.5*err[8],0) + lorentzian(fft_freq_interp,ν3,a3,.5*err[12],0)

    least_square = np.sum(res**2)

    table = tabulate([[str(R2(t0))+' ± '+str(R2(err[0])), str(R2(σ))+' ± '+str(R2(err[1])), str(R2(a_exp))+' ± '+str(R2(err[2])), str(R2(t_exp))+' ± '+str(R2(err[3])), \
                   str(R2(a1))+' ± '+str(R2(err[4])), str(R2(ν1*1000))+' ± '+str(R2(err[5]*1000)), str(R2(ϕ1))+' ± '+str(R2(err[6])), str(R2(a2))+' ± '+str(R2(err[7])), str(R2(ν2*1000))+' ± '+str(R2(err[8]*1000)), str(R2(ϕ2))+' ± '+str(R2(err[9])), \
                   str(R2(t2))+' ± '+str(R2(err[10])), str(R2(a3))+' ± '+str(R2(err[11])), str(R2(ν3*1000))+' ± '+str(R2(err[12]*1000)), str(R2(ϕ3))+' ± '+str(R2(err[13])), str(R2(t3))+' ± '+str(R2(err[14])), str(R2(y0))+' ± '+str(R2(err[15])), *el_cells(el, err[16:]), str(R4(least_square))]],
                   headers=['t0 [ps]', 'σ [ps]', 'a_exp', 'τ [ps]', 'a1', 'ν1 [GHz]', 'ϕ1', 'a2', 'ν2 [GHz]', 'ϕ2', 'τ2 [ps]', 'a3', 'ν3 [GHz]', 'ϕ3', 'τ3 [ps]', 'y0', *el_headers(), 'least square'], tablefmt='orgtbl')
    print(); print(table); print()

    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122);

    ax1.plot(x,y,'--or',lw=2,ms=8,mfc='w')
    ax1.plot(x,fit,'-k',lw=2)
    ax1.plot(x_int,fit_int,'-k',lw=.5)
    ax1.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1.5)
    ax1.plot(x_int,fit_thermal,'-b',lw=.5);
    if plot_Δt == 1:
        ax1.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax1.axhline(y=0,lw=.5,color='k'); ax1.axvline(x=0,lw=.5,color='k')
    ax1.set_xlim([-5,50])
    ax1.set_xlabel('Δt [ps]',fontsize=20); ax1.set_ylabel('TG signal [a.u.]',fontsize=20);
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax2.plot(x,y,'-r',lw=2,label='TG signal')
    # ax2.plot(x,fit,'-k',lw=2)
    ax2.plot(x_int,fit_int,'-k',lw=1,label='Fit')
    ax2.plot(x_int,el_curve(x_int,t0,σ,el),'-g',lw=1,label=el_label())
    ax2.plot(x_int,fit_thermal,'-b',lw=1,label='Thermal decay');
    if plot_Δt == 1:
        ax2.fill_between(x_int,fit_l,fit_u,color='b',alpha=0.2)
    ax2.axhline(y=0,lw=.5,color='k'); ax2.axvline(x=0,lw=.5,color='k')
    ax2.set_xlabel('Δt [ps]',fontsize=20); ax2.set_ylabel('TG signal [a.u.]',fontsize=20); ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig2 = _figure((30, 8))
    ax3 = fig2.add_subplot(131); ax4 = fig2.add_subplot(132); ax5 = fig2.add_subplot(133)

    ax3.plot(x,res*100,'--k',label='residual')
    ax3.plot(x_cut,res_cut*100,'-r',lw=2, label='residual')
    ax3.hlines(y=0, xmin = min(x), xmax = max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual [%]',fontsize=20); ax3.set_xlabel('Δt [ps]',fontsize=20); ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    ax4.plot(fft_freq*1000, fft_res*100, '--b',linewidth=1.5, label='FFT residual')
    ax4.set_xlim([-1,fft_xlim])
    ax4.set_ylabel('FFT amplitude [a.u.][%]',fontsize=20); ax4.set_xlabel('ν [GHz]',fontsize=20); ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    ax5.plot(fft_freq*1000, NormalizeData(fft_res_exp), '--b',linewidth=1.5, label='FFT residual')
    ax5.plot(np.linspace(np.min(fft_freq), np.max(fft_freq), len(fit_lor_tot))*1000,NormalizeData(fit_lor_tot),'-r',lw=2)
    ax5.plot(fft_freq_interp*1000,NormalizeData(fit_lor_tot_rec),'-k',lw=2)
    ax5.set_xlim([-1,fft_xlim])
    ax5.set_ylabel('FFT amplitude [a.u.]',fontsize=20); ax5.set_xlabel('ν [GHz]',fontsize=20); ax5.legend(fontsize=14)
    ax5.tick_params(axis='both', which='major', labelsize=20)

    return dict(fit=fit, popt=popt, err=err, names=FREQ3_DUMP2_NAMES + EL_NAMES, table=table,
                figures=[fig, fig2], title='3 freq 2 dump + ' + el_title(),
                x_int=x_int, fit_int=fit_int, fit_thermal=fit_thermal, res=res,
                fft_freq=fft_freq, fft_res=fft_res, least_square=least_square)


GLOBAL_MODELS = {
    # n frequencies: (function, parameter names, indices of damping times)
    1: (freq1_dump1, FREQ1_DUMP1_NAMES),
    2: (freq2_dump1, FREQ2_DUMP1_NAMES),
    3: (freq3_dump2, FREQ3_DUMP2_NAMES),
}

GLOBAL_MODELS_EL = {
    1: (freq1_dump1_el, FREQ1_DUMP1_NAMES + EL_NAMES),
    2: (freq2_dump1_el, FREQ2_DUMP1_NAMES + EL_NAMES),
    3: (freq3_dump2_el, FREQ3_DUMP2_NAMES + EL_NAMES),
}


def guess_global(n, thermal_popt, thermal_cons, lor_popt, df=0.001, spans_ghz=None, centres=None, el=False):
    """p0/cons for the global n-frequency fit.

    t0, σ, a_exp, τ, y0 start from the thermal fit (with its bounds);
    each ν starts from the lorentzian fit of the FFT, bounded to
    centre ± span (centres: THz, default the lorentzian ν; spans_ghz: see
    freq_windows, default ±max(3 GHz, 2 bins));
    amplitudes 0.3 in (0.01, 1), phases 0.1 in (-3.14, 3.14), damping
    times 100 ps in (1, 5000), as in the notebook cells.
    """
    names = (GLOBAL_MODELS_EL if el else GLOBAL_MODELS)[n][1]
    t0, σ, a_exp, t_exp, y0 = thermal_popt[:5]
    lo_t, hi_t = thermal_cons
    nus = [lor_popt[3*i] for i in range(n)]
    cen = list(centres or [])[:n]
    cen += nus[len(cen):]
    wins = freq_windows(n, spans_ghz, df)
    p0, lo, hi = [], [], []
    for name in names:
        if name in EL_NAMES:   # electronic peak: from the thermal + electronic fit when there was one
            k = EL_NAMES.index(name)
            if len(thermal_popt) > 5:
                p0.append(thermal_popt[5 + k]); lo.append(lo_t[5 + k]); hi.append(hi_t[5 + k])
            else:
                p0.append(EL_P0[k]); lo.append(EL_CONS[0][k]); hi.append(EL_CONS[1][k])
        elif name in ('t0', 'σ', 'a_exp', 'τ', 'y0'):
            k = THERMAL_NAMES.index(name)
            v = (t0, σ, a_exp, t_exp, y0)[k]
            p0.append(v); lo.append(lo_t[k]); hi.append(hi_t[k])
        elif name.startswith('a'):
            p0.append(0.3); lo.append(0.01); hi.append(1.0)
        elif name.startswith('ν'):
            k = int(name[1])-1
            p0.append(nus[k]); lo.append(max(cen[k]-wins[k], 0.0)); hi.append(cen[k]+wins[k])
        elif name.startswith('ϕ'):
            p0.append(0.1); lo.append(-3.14); hi.append(3.14)
        elif name.startswith('τ'):
            p0.append(100.0); lo.append(1.0); hi.append(5000.0)
    return _clip(p0, lo, hi), (tuple(lo), tuple(hi))


def _clip(p0, lo, hi):
    out = []
    for v, l, h in zip(p0, lo, hi):
        if not l < h:
            raise ValueError(f"lower bound {l} not below upper bound {h}")
        span = h - l
        out.append(float(min(max(v, l + 1e-9*span), h - 1e-9*span)))
    return tuple(out)


def run_global(n, x, y, fft_res_exp, fit_lor_tot, cut, fft_xlim, p0, cons, plot_dt=1, el=False):
    func = (GLOBAL_MODELS_EL if el else GLOBAL_MODELS)[n][0]
    if n == 1:
        return func(x, y, fft_res_exp, fit_lor_tot, cut, fft_xlim, p0, cons)
    return func(x, y, fft_res_exp, fit_lor_tot, cut, fft_xlim, plot_dt, p0, cons)


# --------------------------------------------------------------------------
# comparison of several global fits in the same axes
# --------------------------------------------------------------------------

COMPARE_COLORS = {1: 'g', 2: 'b', 3: 'm'}


def compare_fits(x, y, results, fft_xlim):
    """Overlay several global fits: fit (zoom and full trace), residual and
    FFT of the residual, one colour per model, same layout as the notebook
    figures. results: {n_freq: dict returned by run_global}."""
    fig = _figure((30, 8))
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
    ax1.plot(x, y, '--or', lw=2, ms=8, mfc='w', label='TG signal')
    ax2.plot(x, y, '-r', lw=2, label='TG signal')
    for n, g in sorted(results.items()):
        c = COMPARE_COLORS.get(n, 'k')
        lab = f"{g['title']} (LS {R4(g['least_square'])})"
        ax1.plot(g['x_int'], g['fit_int'], '-', color=c, lw=2.5, label=lab)
        ax2.plot(g['x_int'], g['fit_int'], '-', color=c, lw=2.5, label=lab)
    for ax in (ax1, ax2):
        ax.axhline(y=0, lw=.5, color='k'); ax.axvline(x=0, lw=.5, color='k')
        ax.set_xlabel('Δt [ps]', fontsize=20); ax.set_ylabel('TG signal [a.u.]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20); ax.legend(fontsize=16)
    ax1.set_xlim([-5, 50])

    fig2 = _figure((30, 8))
    ax3 = fig2.add_subplot(121); ax4 = fig2.add_subplot(122)
    for n, g in sorted(results.items()):
        c = COMPARE_COLORS.get(n, 'k')
        ax3.plot(x, g['res']*100, '-', color=c, lw=2.5, label=g['title'])
        ax4.plot(g['fft_freq']*1000, g['fft_res']*100, '-', color=c, lw=2.5, label=g['title'])
        for k in range(1, n+1):
            nu = g['popt'][g['names'].index(f'ν{k}')]
            ax4.axvline(x=nu*1000, color=c, lw=2, ls='--')
    ax3.hlines(y=0, xmin=min(x), xmax=max(x), linewidth=1, color='k')
    ax3.set_ylabel('Residual [%]', fontsize=20); ax3.set_xlabel('Δt [ps]', fontsize=20)
    ax4.set_xlim([-1, fft_xlim])
    ax4.set_ylabel('FFT residual amplitude [a.u.][%]', fontsize=20); ax4.set_xlabel('ν [GHz]', fontsize=20)
    for ax in (ax3, ax4):
        ax.tick_params(axis='both', which='major', labelsize=20); ax.legend(fontsize=14)
    return [fig, fig2]


# --------------------------------------------------------------------------
# several scans: read, merge (weighted average), overlay
# --------------------------------------------------------------------------

def read_scan(path):
    """(time, signal, err or None) of a Scan_NNN_nor.txt, sorted in time.
    err comes from Scan_NNN_err.txt next to it (same time points), if there."""
    d = np.atleast_2d(np.loadtxt(path))
    t, y = d[:, 0].astype(float), d[:, 1].astype(float)
    e = None
    ep = path[:-len('_nor.txt')] + '_err.txt' if path.endswith('_nor.txt') else None
    if ep:
        try:
            de = np.atleast_2d(np.loadtxt(ep))
            if len(de) == len(t) and np.allclose(de[:, 0], t):
                e = de[:, 2].astype(float)
        except (OSError, ValueError):
            pass
    o = np.argsort(t, kind='stable')
    return t[o], y[o], (None if e is None else e[o])


def merge_scans(scans, tol=None, scale=False):
    """Weighted average of repeated scans.

    scans: list of (t, y, err or None). Time points closer than tol (default a
    quarter of the smallest time step) are the same delay. With errors on
    every scan: y = Σ(y/σ²)/Σ(1/σ²), σ = 1/sqrt(Σ 1/σ²); otherwise the plain
    mean, σ = std/sqrt(n) (NaN for a delay measured once).
    scale=True first multiplies each scan by the factor that best matches it
    to the first one (weighted least squares on the delays they share).
    Returns t, y, err, std between scans, number of scans per point, factors."""
    scans = [(np.asarray(t, float), np.asarray(y, float), None if e is None else np.asarray(e, float))
             for t, y, e in scans]
    if tol is None:
        steps = np.concatenate([np.diff(np.unique(t)) for t, _, _ in scans] + [np.array([np.inf])])
        steps = steps[steps > 0]
        tol = 0.25*np.min(steps) if np.isfinite(np.min(steps)) else 1e-6
    weighted_ok = all(e is not None for _, _, e in scans)
    factors = [1.0]*len(scans)
    if scale and len(scans) > 1:
        t0, y0, _ = scans[0]
        for k in range(1, len(scans)):
            t, y, e = scans[k]
            j = np.searchsorted(t0, t)
            near = [(i, jj) for i, jj in enumerate(j) for jj in (jj - 1, jj) if 0 <= jj < len(t0)
                    and abs(t0[jj] - t[i]) <= tol]
            if len(near) < 2:
                continue
            i, jj = np.array(near).T
            w = 1/clean_err(e)[i]**2 if e is not None else np.ones(len(i))
            den = np.sum(w*y[i]**2)
            if den > 0:
                factors[k] = float(np.sum(w*y0[jj]*y[i])/den)
    allt = np.concatenate([t for t, _, _ in scans])
    ally = np.concatenate([y*f for (_, y, _), f in zip(scans, factors)])
    alle = np.concatenate([(clean_err(e) if e is not None else np.full(len(t), np.nan))*abs(f)
                           for (t, _, e), f in zip(scans, factors)])
    o = np.argsort(allt, kind='stable')
    allt, ally, alle = allt[o], ally[o], alle[o]
    groups, cur = [], [0]
    for k in range(1, len(allt)):
        if allt[k] - allt[cur[0]] <= tol:
            cur.append(k)
        else:
            groups.append(cur); cur = [k]
    groups.append(cur)
    T, Y, E, S, N = [], [], [], [], []
    for g in groups:
        tt, yy, ee = allt[g], ally[g], alle[g]
        n = len(g)
        if weighted_ok:
            w = 1/ee**2
            Y.append(np.sum(w*yy)/np.sum(w)); E.append(1/np.sqrt(np.sum(w)))
        else:
            Y.append(np.mean(yy)); E.append(np.std(yy, ddof=1)/np.sqrt(n) if n > 1 else np.nan)
        T.append(np.mean(tt)); S.append(np.std(yy, ddof=1) if n > 1 else 0.0); N.append(n)
    return (np.array(T), np.array(Y), np.array(E), np.array(S), np.array(N), factors)


def write_merged(path_nor, t, y, err, std, n, sources, factors):
    """Merged trace in the same layout as the scan outputs: _nor.txt
    (time, signal, std between scans) and _err.txt (time, signal, error,
    std, scans per point), so the trace analysis reads it like any scan."""
    fmt = '%.7e'
    head = 'merged from:\n' + '\n'.join(f'{p}  (x {f:.4g})' for p, f in zip(sources, factors))
    np.savetxt(path_nor, np.column_stack([t, y, std]), fmt=fmt, delimiter='\t')
    np.savetxt(path_nor[:-len('_nor.txt')] + '_err.txt', np.column_stack([t, y, err, std, n]), fmt=fmt,
               delimiter='\t', header=head + '\ntime\tTGSignal\terror\tstd between scans\tscans per point')
