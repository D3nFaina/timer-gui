# TIMER GUI

## Start it from Spyder
1. Anaconda Navigator, then Spyder.
2. Open `Software/python/timer_gui.py` and press **Run (F5)**. You can also type `%run timer_gui.py` in the console from this folder.
3. The window blocks the console until you close it. If Spyder complains about the event loop, go to Run, then Configuration per file, and choose **Execute in a dedicated console**.

You need numpy, scipy, matplotlib and h5py, which all ship with Anaconda. `tabulate` is optional: it gives nicer tables, and there is a built-in fallback without it.

## Main window: scan analysis
This is a port of `AnalisiMaxiTimer200505*.m`.
- **Scan tab**: experiment folder, scan number and name format (`Scan_%03.0f`), OnlyProbe and NoProbe folder suffixes (`_OnlyProbe`, `_NoProbe`), x axis (FEL delay, seed delay line 2, phase shifters, M2Z), T0, bin, file range, excluded files, I0 threshold and normalisation (I0³, I0²·spectrum or I0·spectrum).
- **ROI / spectrum tab**: ROI `[col, row, width, height]`, ellipse axes and centre (1-based, the same numbers as in the MATLAB input section), how the centre is found (fixed, maximum or barycentre), proximity background frame, and spectrometer window.
- **Options**:
  - *Subtract background*: four choices.
    - *OnlyProbe file, nearest bunch*: subtracts the `Scan_NNN_OnlyProbe` image (probe only, no FEL) whose bunch number is closest to the signal file. Removes probe scatter. MATLAB: `BG_Sig=0`, `BGStr='_OnlyProbe'`.
    - *NoProbe file, nearest bunch*: subtracts the `Scan_NNN_NoProbe` image (FEL and pumps, no probe). Removes FEL/pump scatter. MATLAB: `BG_Sig=0`, `BGStr='_NoProbe'`.
    - *NoProbe + OnlyProbe*: subtracts both. The camera offset is then subtracted twice; the proximity background removes it again.
    - *Mean of the first N signal images*: MATLAB `BG_Sig=N`. On SrO Scan_002 this gave the cleanest trace.
    - Note: MATLAB's `GetBGbyBunchNumber.m` reads the bunch number from the wrong part of the current file names (`Scan_002_<bunch>.h5`), so MATLAB always used the first background file. The GUI reads the real bunch number.
  - *Single image only*: analyses one file and shows the full image with the ROI and the zoom with the ellipse, so you can place the ROI.
  - *Loop*: re-reads the folder every N seconds and adds the new files. The last 3 points are redone each time, as in MATLAB.
- *Fast run*: ticked = as fast as possible (plots are drawn at the end). Unticked = every file is drawn and the run pauses on it for the seconds in the box next to it (default 0.3; arrows or typing). Both the tickbox and the pause can be changed during a run. The redraw itself takes a fraction of a second, so very short pauses are limited by it. *TG signal vs file* also shows the delay on a top axis.
- Every 2D image (full image, zoom, background, timing) has a colorbar in CCD counts per pixel. The full and zoom images are lightly smoothed for display (and the full image is averaged 2×2), so the colorbar shows smoothed counts; the analysis uses the raw pixels.
- **Stop (save)** finishes the current file, then bins and saves everything analysed so far. **Abort (no save)** stops and writes nothing.
- **Tabs on the right**:
  - *Scan* holds the MATLAB figures: IoM Sh, spectrum, full image with the ROIs and zoom with the ellipse, both linear and contrast-enhanced for display only (smoothed 1.5 px, colours from the median to the 99.95th percentile, STRETCH_* in timer_gui.py), zoom with the ellipse, TG vs file, TG vs time, signal shift, binned TG, background variation.
  - *I0 / spectrometer statistics* holds the good shots of every file analysed so far, and grows with each new file: histograms of I0 and of the spectrometer integral (all files plus the current one), per-file mean ± std against the overall mean ± 1σ, I0 against spectrometer-integral scatter with its correlation, and the fraction of good shots per file.
  - *Background* holds the raw spectrum with the normalisation window and baseline windows marked, the background image subtracted from the current file (ROI), the proximity background per file, and the background-file stability check (MATLAB figures 23 and 24).
- **Error bands** (shaded, `fill_between`) on *TG signal vs file*, *TG signal* and the binned trace. Each file is one integrated CCD image, so the error is propagated per file:
  - pixel noise σ_px = pooled std of the background-subtracted image in the four proximity frames (each frame around its own mean);
  - σ²(TG sum) = N_ell·σ_px² + (N_ell·σ_prbg)² + g·TG sum, with σ_prbg the error of the subtracted proximity background and g the *CCD counts per photon* (ROI tab, 0 = no signal shot noise);
  - normalisation: relative standard error of mean(I0³), mean(I0²)·mean(S) or mean(I0)·mean(S) over the good shots, I0–spectrometer covariance included, added in quadrature;
  - *CCD counts per photon* defaults to 36, measured on SrO Scan_002 (1σ range 27–47). *Estimate counts per photon from the last scan* fits it to the scatter of each point around a local quadratic of its neighbours (points after t = 3 ps, at least 10 needed). Anything that grows with the signal ends up in it, so it is an effective value; run the scan again to apply a new one.
  - binning: σ_bin = sqrt(Σσ_i²)/n. The MATLAB std column (scatter between files of a bin) is unchanged.
  - Written to `Scan_NNN_err.txt`: time, TG signal, propagated error, std between files in the bin, files in the bin. The MATLAB files are unchanged.
  - The Trace analysis window reads `_err.txt` next to the `_nor.txt` file and shades the same band (scaled like the normalised trace). The fits themselves are unweighted, as in the notebook.
- Results go to `<output>/<Experiment>/Scan_NNN_raw/_nor/_pos/_bg/_MNF.txt`, with the same columns as MATLAB, plus a `.mat` file and `_parameters.json`. The default output folder is `<parent of experiment folder>/results/OutPut`.
- The last settings are remembered in `~/.timer_gui_settings.json`. *Save/Load settings* keeps one file per sample.

## Timing (Svirgo) window
This is a port of `Timing/Svirgo_comparison.m`, and you use it at the start of the beamtime.
- Set the pump A and pump B scan folders, the pump trajectory (1 to 4) and the FEL-off threshold. *Auto* sets the threshold from the I0 histogram.
- **Number of ROIs**: choose 1 (normal) or 2 (Wollaston prism). Type the ROIs in, or use *Draw ROIs on the image*.
- *Draw ROIs on the image* now shows the FEL-on minus FEL-off image (jet), where the spot is visible.
- *Find the peak: automatic ROI*: builds the FEL-on minus FEL-off image from the files with the strongest spot, finds the peak, and follows 5 parallel cuts in x and in y outwards until the spot falls to the background level (the larger of 3× the noise and 10% of the peak). The median edge plus 1 pixel is the ROI. A window shows the image with the ROI and the two averaged cuts with the ROI limits as vertical lines. With 2 ROIs the second spot is searched after masking the first.
- *Save pictures and traces*: writes timing_traces.png, timing_roi.png and timing_pumpA/B.txt (delay, ROI variation in %) to a folder you choose.
- *Auto update* polls both folders. When A and B have the same number of points, it cross-correlates them, then prints dT(A-B) and the suggested pump Z position.

## Trace analysis window
This uses the fit models from `SiO2_Baldi_Analysis.ipynb`, with the same figures.
- Load a `_nor.txt` file (it defaults to the last scan analysed). Set the time shift, the baseline points used for subtraction after normalisation, the crop range and the FFT `cut`.
- The thermal decay start values and bounds can be edited.
- Pick any of: *Thermal only*, *1 frequency* (1 freq 1 dump), *2 frequencies* (2 freq 1 dump), *3 frequencies* (3 freq 2 dump, frequencies 2 and 3 damped).
- *Thermal filter*: the thermal fit is rejected and not drawn when t0, σ, a_exp or τ ends on a bound, when the τ ± error band opens wider than the set percentage of the signal (default 100%), or when R² is below the set minimum (default 0.3). The reasons are printed. Tick *Plot the thermal fit even when rejected* to draw it anyway.
- *Uneven time steps* (for example 0.2 ps around t0 and 2 ps later): the step sections are listed, and every FFT is computed on the real time points rather than assuming the first step. Uniform scans give the same numbers as the notebook FFT.
- Always shown: the trace with the thermal decay, and the FFT of the residual with its Lorentzian fit. The Lorentzian frequencies are the starting guesses for the global fits; you can also type ν1..ν3 in GHz. *Frequency span ±Δν1, Δν2, Δν3* limits each frequency to ν ± Δν (GHz) in both the Lorentzian and the global fits, like ν1_l = ν1 − Δν1, ν1_u = ν1 + Δν1 in the notebook. One number applies to all frequencies; blank = ±3 GHz. The bounds are centred on your typed ν, or on the FFT peak when ν is blank, and show up in *Edit guesses and refit*.
- One frequency box ticked: the full fit and the FFT of its residual, as in the notebook. Several boxes ticked: the fits and the residual FFTs are overlaid in the same axes, one colour per model.
- *Edit guesses and refit* opens the start/lower/upper table of each global fit, filled with the last fitted values.

## Files
- `timer_io.py`: HDF5 loading (from the loader thread).
- `timer_analysis.py`: scan analysis engine, which also works without the GUI.
- `timer_timing.py`: Svirgo timing engine.
- `trace_fit.py`: Den's fit functions.
- `timer_gui.py`: the GUI.

## Added in the last round
- **Weighted fits** (trace window, tickbox *Weighted fit*): the thermal and global fits weight each point by 1/error² from `Scan_NNN_err.txt`. Unticked = unweighted, exactly as the notebook. Every fit table now ends with the reduced χ²: about 1 means the curve is as good as the error bars allow, much more than 1 means something is missing from the model (e.g. the thermal-only fit of an oscillating trace). With weights the parameter errors come from the error bars.
- **Barycentre fixed.** Barycenter.m weighted pixels by (max − I), so the darkest pixels counted most and the centre sat near the middle of the box whatever the spot did. Now the weights are the smoothed intensity above half way between the background (median) and the peak. Only `_pos.txt` (cg, rg) changes, plus the ellipse position when *Ellipse centre* is *barycenter*. The tickbox *Barycentre as old MATLAB* in the ROI tab brings the old behaviour back, to reproduce old files: with it ticked every output is still identical to MATLAB.
- **Several scans** (trace window, box on the right): add `_nor.txt` files to the list (the `_err.txt` next to each is read automatically; picking an `_err.txt` adds its `_nor.txt`).
  - *Merge* makes a weighted average (1/error²; a plain mean if some scan has no `_err.txt`), writes `Merged_Scan_002+003_nor.txt` and `_err.txt` next to the first scan, shows the inputs with the result and makes it the trace to fit. Delays closer than a quarter of the smallest step count as the same delay. *Scale each scan to the first one* first multiplies each scan by its least-squares factor (useful when the FEL flux or alignment changed between scans).
  - *Overlay* plots all listed traces in one figure with their error bands, normalised (0 = mean of the first N points, 1 = peak; or in counts per shot). *Overlay baseline* takes one N for all traces, or a list N1, N2, … in the order of the list, the last value used for any further trace (5, 4 = 5 for the first trace, 4 for all the others), with an optional vertical offset.
- **PDF reports.** Main window *PDF report*: `Scan_NNN_report.pdf` next to the output files, with the settings and the Scan, statistics and background tabs. Trace window *PDF report…*: settings, every fit table and all the figures of the last run (plus the last merge and overlay). The settings of every scan are also saved as `Scan_NNN_parameters.json` (loadable with *Load settings…*).

- **Coherent spike / electronic peak** (trace window, box next to the thermal guesses): ticked, every fit (thermal, 1, 2 and 3 frequencies) gets a fast term, in one of two forms:
  - **Coherent spike** (default; resonant pumping, circular pumps, linear probe): I = |I_TG ± B·G|² + |C·G|², with G = exp(−((t−t_cs)/σ_cs)²). B·G interferes with the TG field (the sign of B decides + or −); C·G is the orthogonal polarisation, with its own amplitude, and only adds. Parameters: B in [−3, 3], C in [0, 3], t_cs in [−2, 15] ps, σ_cs in [5, 500] fs. G has its own centre and width because the erf of I_TG now comes from the spike-free thermal fit, where σ is the slow build-up of the grating (magnetite: 0.6 ps), not the time resolution. Where the TG field is small at the spike, B² and C² do the same job, so their errors come out large and correlated; their sum is well defined.
  - **Free peak**: |I_TG + a_el·P(t)|², P a step at its own start t_el with rise σ_el, decaying with τ_el (an exponential convolved with the Gaussian pulse, height ≤ 1). Use it when the TG part rises more slowly than the spike. Bounds: a_el in [−3, 3], t_el in [−2, 15] ps, σ_el in [5, 200] fs, τ_el in [5, 2000] fs.
  - The thermal fit leaves the spike out: points between 0.1 ps before the steepest rise and the first minimum after the spike (or the window typed in 'Thermal fit skips the spike, t from, to') are not fitted, shown as orange crosses on a yellow band, and are taken out of the residual that goes to the FFT. The fast term enters only the 1, 2 and 3 frequency fits.
  - Blank start boxes (B, C, a_el, t_el) come from the data: the height and steepest rise of the spike. The fast term alone is drawn dashed in green. With 'Frequency fits keep t0, σ, a_exp, τ, y0 of the thermal fit' ticked (default), those five are held at the thermal-fit values (shown ± 0) and only the oscillations and the fast term are fitted; unticked, they start there and are free. Unticked, the fits are exactly the notebook ones.
  - Magnetite Scan_035, thermal fixed (t0 0.82 ps, σ 0.62 ps, τ 30 ps): coherent spike LS 0.92 / 0.23 / 0.22 for 1 / 2 / 3 frequencies (t_cs 0.14 ps, σ_cs 98 fs); free peak 0.81 / 0.09.
- **Single image: move the ellipse with the mouse.** In the zoom image, drag the ellipse to move it; scroll changes its width, shift + scroll its height. On release the 'Ellipse centre in ROI' and 'Ellipse axes' fields are updated and the centre mode is set to fixed, so the next run (and the saved Scan_NNN_parameters.json) uses them. The toolbar's zoom/pan tools take the mouse while they are on.
- **Files without a CCD image** (e.g. Scan_037_185340809.h5, saved without the camera) are skipped with a line in the log, as MATLAB's try/catch does. A background file without an image is replaced by the next nearest one, the first-N background takes the next files that have one, and single-image mode shows the next file.
- **FFT peak search** stops at the "FFT plot limit" (default 120 GHz). Scans with a finely sampled stretch near t0 (0.01 ps steps) used to pick THz noise peaks as ν2, ν3.
