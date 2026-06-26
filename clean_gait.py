"""
Gait Cleaning Pipeline for DLC Bipedal Treadmill Data
======================================================
Processes DeepLabCut pose-estimation CSVs from a mouse bipedal treadmill task.

Pipeline Stages
---------------
1.  Parse multi-row DLC header → tidy DataFrame
2.  Per-bodypart likelihood masking (low-confidence → NaN)
3.  Outlier rejection: velocity spike removal (tracking jumps)
4.  Short-gap interpolation (linear, ≤ max_gap frames)
5.  Bout detection via composite motion score:
      - ankle + foot x-velocity weighted by likelihood
      - rolling median smoothing + hysteresis to find sustained motion regions
      - close inactive gaps are merged so W06-style recovery stepping is not
        split into many tiny fragments
6.  Bout quality filter:
      - minimum duration
      - require periodic x-signal (sawtooth = treadmill stepping)
7.  Within-bout Savitzky-Golay smoothing
8.  Export: cleaned CSV, bout metadata CSV, 5 diagnostic plots

Bout metadata includes duration, fraction of total recording time, estimated
step count, and step frequency (steps/second).

Stepping vs Dragging
--------------------
Both appear as the same treadmill sawtooth in x.  We label each bout
with a "step_quality" score (0–1) based on the regularity of the x-cycle:
  - High score  → rhythmic stepping (clear periodicity)
  - Low score   → dragging / irregular motion
This score is included in cleaned_gait_data.csv so downstream analysis
can stratify bouts.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, welch
from scipy.ndimage import label, binary_dilation, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os

# ─────────────────────────── CONFIG ────────────────────────────
INPUT_DIR = Path(os.environ.get("GAIT_INPUT_DIR", "./Week6"))
if not INPUT_DIR.exists():
    INPUT_DIR = Path(".")
OUT_DIR = Path(os.environ.get("GAIT_OUT_DIR", str(INPUT_DIR / "Cleaned")))

INPUT_FILES = sorted(
    p for p in INPUT_DIR.glob("*.csv")
    if not p.name.endswith("_cleaned_gait_data.csv")
    and p.name != "cleaned_gait_data.csv"
    and not p.name.endswith("_bout_metadata.csv")
)

if not INPUT_FILES:
    print(f"No raw DLC CSV files found in {INPUT_DIR.resolve()}")

for INPUT_CSV in INPUT_FILES:
    print(f"Processing: {INPUT_CSV.name}")

    # Likelihood filtering  ─────────────────────────────────────────
    LIKELIHOOD_THRESH = {
        "back":   0.55,   # back is harder to track; lower threshold
        "pelvis": 0.50,
        "knee":   0.55,
        "ankle":  0.60,
        "foot":   0.55,
    }
    
    # Velocity spike rejection  ────────────────────────────────────
    # Frame-to-frame displacement > this value is treated as a tracking jump
    VELOCITY_SPIKE_THRESH = 30.0    # pixels/frame
    
    # Gap interpolation  ───────────────────────────────────────────
    MAX_INTERP_GAP = 5              # max consecutive NaN frames to fill
    
    # Bout detection  ──────────────────────────────────────────────
    # Composite motion score = weighted sum of ankle + foot x-velocity
    # (x captures treadmill belt direction; y captures step height)
    FRAME_RATE           = 15.10    # frames/second; used for bout duration and step frequency
    MOTION_ROLL_WINDOW   = 11       # rolling median window for smoothing signal
    MOTION_THRESH        = 2.5      # high threshold: starts/anchors an active stepping bout
    MOTION_OFF_THRESH    = 1.25     # low threshold: keeps an existing bout alive through dips
    MOTION_DILATION_FRAMES = 3      # expand active mask by N frames each side
    MAX_INACTIVE_GAP_FRAMES = 30    # merge bouts separated by <= this many inactive frames
    MIN_BOUT_FRAMES      = 30       # discard very short fragments after merging
    MIN_STEPS_PER_BOUT   = 0        # set >0 to discard fragments with too few step cycles
    MIN_STEP_INTERVAL_SEC = 0.18    # fastest plausible interval between same-foot cycles
    HIGH_CONFIDENCE_MEAN_LIKELIHOOD = 0.95

    # Legacy settings are kept for lower-confidence W00-style files where the
    # original, more local bout splitting was already useful.
    LEGACY_MOTION_ROLL_WINDOW = 7
    LEGACY_MOTION_OFF_THRESH = MOTION_THRESH
    LEGACY_MOTION_DILATION_FRAMES = 5
    LEGACY_MAX_INACTIVE_GAP_FRAMES = 0
    LEGACY_MIN_BOUT_FRAMES = 20
    LEGACY_MIN_STEPS_PER_BOUT = 0
    
    # Step quality scoring  ────────────────────────────────────────
    # Spectral regularity of foot x signal within a bout
    # ≥ this fraction of power in top frequency band → "stepping" quality
    STEP_QUALITY_BAND_FRAC = 0.30   # dominant freq must hold ≥30% of total power
    
    # Smoothing  ───────────────────────────────────────────────────
    SG_WINDOW  = 11
    SG_POLY    = 3
    
    # ──────────────────────────────────────────────────────────────
    
    
    # ── 1. PARSE ───────────────────────────────────────────────────
    def parse_dlc_csv(path: Path):
        raw = pd.read_csv(path, header=None)
        bodyparts = raw.iloc[1, 1:].values
        coords    = raw.iloc[2, 1:].values
        cols = [f"{bp}_{c}" for bp, c in zip(bodyparts, coords)]
        df = raw.iloc[3:].copy()
        df.columns = ["frame"] + cols
        df = df.reset_index(drop=True)
        df["frame"] = df["frame"].astype(int)
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        parts = list(dict.fromkeys(bodyparts))
        return df, parts
    
    
    # ── 2. LIKELIHOOD FILTER ───────────────────────────────────────
    def apply_likelihood_filter(df: pd.DataFrame, parts: list, thresh: dict) -> pd.DataFrame:
        df = df.copy()
        for p in parts:
            t = thresh.get(p, 0.6)
            mask = df[f"{p}_likelihood"] < t
            df.loc[mask, f"{p}_x"] = np.nan
            df.loc[mask, f"{p}_y"] = np.nan
        return df
    
    
    # ── 3. VELOCITY SPIKE REJECTION ────────────────────────────────
    def reject_velocity_spikes(df: pd.DataFrame, parts: list, spike_thresh: float) -> pd.DataFrame:
        """
        If a bodypart jumps > spike_thresh px in one frame in either axis,
        null out that frame (likely a tracking dropout or swap).
        """
        df = df.copy()
        for p in parts:
            for coord in ["x", "y"]:
                col = f"{p}_{coord}"
                vel = df[col].diff().abs()
                df.loc[vel > spike_thresh, col] = np.nan
        return df
    
    
    # ── 4. GAP INTERPOLATION ──────────────────────────────────────
    def interpolate_gaps(df: pd.DataFrame, parts: list, max_gap: int) -> pd.DataFrame:
        df = df.copy()
        for p in parts:
            for coord in ["x", "y"]:
                col = f"{p}_{coord}"
                s = df[col].copy()
                is_na = s.isna()
                gap_id = is_na.ne(is_na.shift(fill_value=False)).cumsum()
                gap_sizes = is_na.groupby(gap_id).transform("sum")
                interp = s.interpolate(method="linear", limit=max_gap,
                                       limit_direction="forward")
                df[col] = interp
                df.loc[is_na & (gap_sizes > max_gap), col] = np.nan
        return df
    
    
    # ── 4b. LIKELIHOOD NORMALISATION ─────────────────────────────
    def normalize_likelihoods(df: pd.DataFrame, parts: list) -> pd.DataFrame:
        """
        Normalize each body-part's likelihood column to a [0, 1] range using
        per-part min-max scaling so that parts with consistently high raw
        likelihoods (e.g. marker-aided keypoints) don't dominate likelihood-
        weighted calculations relative to harder-to-track keypoints.
    
        Normalised value = (lk - part_min) / (part_max - part_min)
        A small epsilon prevents division-by-zero when the range is flat.
        """
        df = df.copy()
        for p in parts:
            col = f"{p}_likelihood"
            raw = df[col]
            lo, hi = raw.min(), raw.max()
            rng = hi - lo
            if rng < 1e-9:
                # All values identical — set to 1.0 (always trusted equally)
                df[col] = 1.0
            else:
                df[col] = (raw - lo) / rng
        return df
    
    
    # ── 5. BOUT DETECTION ──────────────────────────────────────────
    def build_motion_score(df: pd.DataFrame) -> pd.Series:
        """
        Composite motion = ankle x-vel (lk-weighted) + foot x-vel (lk-weighted).
        Both x and y components contribute; x dominates on a treadmill.
        Likelihoods are expected to be already normalised per body part so that
        marker-aided and markerless keypoints are on an equal footing.
        """
        ankle_lk = df["ankle_likelihood"].clip(0, 1)
        foot_lk  = df["foot_likelihood"].clip(0, 1)
        ankle_xv = df["ankle_x"].diff().abs().fillna(0)
        foot_xv  = df["foot_x"].diff().abs().fillna(0)
        ankle_yv = df["ankle_y"].diff().abs().fillna(0)
        foot_yv  = df["foot_y"].diff().abs().fillna(0)
        score = (
            ankle_lk * (ankle_xv * 1.0 + ankle_yv * 0.5) +
            foot_lk  * (foot_xv  * 1.0 + foot_yv  * 0.5)
        )
        return score
    
    
    def _hysteresis_active_mask(smooth_score: pd.Series,
                                on_thresh: float,
                                off_thresh: float) -> np.ndarray:
        """
        Start activity at the higher threshold, then keep the bout active until
        the score falls below the lower threshold. This prevents a long W06
        stepping run from being split at every brief low-velocity phase.
        """
        active = np.zeros(len(smooth_score), dtype=bool)
        in_bout = False
        for i, val in enumerate(smooth_score.values):
            if val >= on_thresh:
                in_bout = True
            elif val < off_thresh:
                in_bout = False
            active[i] = in_bout
        return active


    def _merge_short_inactive_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
        """Fill inactive gaps of up to max_gap frames between active regions."""
        if max_gap <= 0 or not active.any():
            return active

        merged = active.copy()
        inactive = ~active
        labeled, n_gaps = label(inactive)
        for i in range(1, n_gaps + 1):
            idx = np.where(labeled == i)[0]
            if len(idx) == 0 or len(idx) > max_gap:
                continue
            has_active_before = idx[0] > 0 and active[idx[0] - 1]
            has_active_after = idx[-1] < len(active) - 1 and active[idx[-1] + 1]
            if has_active_before and has_active_after:
                merged[idx] = True
        return merged


    def _label_bouts(active: np.ndarray, min_frames: int):
        labeled, n_raw = label(active)
        bout_mask = np.zeros(len(active), dtype=bool)
        bout_ids = np.zeros(len(active), dtype=int)
        bout_num = 0
        for i in range(1, n_raw + 1):
            idx = np.where(labeled == i)[0]
            if len(idx) >= min_frames:
                bout_num += 1
                bout_mask[idx] = True
                bout_ids[idx] = bout_num
        return bout_mask, bout_ids


    def detect_bouts(df: pd.DataFrame,
                     motion_thresh: float,
                     motion_off_thresh: float,
                     roll_window: int,
                     dilation: int,
                     max_inactive_gap: int,
                     min_frames: int):
        score = build_motion_score(df)
        smooth_score = score.rolling(roll_window, center=True, min_periods=1).median()

        active = _hysteresis_active_mask(smooth_score, motion_thresh, motion_off_thresh)
        if dilation > 0:
            struct = np.ones(dilation * 2 + 1, dtype=bool)
            active = binary_dilation(active, structure=struct)
            active = binary_erosion(active, structure=np.ones(3, dtype=bool))
        active = _merge_short_inactive_gaps(active, max_inactive_gap)
        bout_mask, bout_ids = _label_bouts(active, min_frames)

        return bout_mask, bout_ids, smooth_score
    
    
    # ── 6. STEP QUALITY SCORING ────────────────────────────────────
    def score_bout_periodicity(df: pd.DataFrame, bout_ids: np.ndarray,
                                band_frac: float) -> dict:
        """
        Compute spectral regularity of ankle x within each bout.
        Returns dict {bout_id: step_quality (0–1)}.
        """
        quality = {}
        for b in np.unique(bout_ids[bout_ids > 0]):
            idx = np.where(bout_ids == b)[0]
            sig = df["ankle_x"].iloc[idx].values
            # fill any remaining NaN
            sig = pd.Series(sig).interpolate(limit_direction="both").ffill().bfill().values
            if len(sig) < 10 or np.all(np.isnan(sig)):
                quality[b] = 0.0
                continue
            # Welch power spectrum
            try:
                nperseg = min(len(sig), 32)
                f, pxx = welch(sig - np.nanmean(sig), nperseg=nperseg)
                if pxx.sum() == 0:
                    quality[b] = 0.0
                else:
                    # fraction of power held by the dominant frequency
                    quality[b] = float(pxx.max() / pxx.sum())
            except Exception:
                quality[b] = 0.0
        return quality


    def count_steps_in_signal(sig: np.ndarray,
                              frame_rate: float,
                              min_interval_sec: float) -> int:
        """
        Count repeated foot x cycles inside a bout. The treadmill signal is
        sawtooth-like, so local maxima in the smoothed foot x trace are a
        practical same-foot step-cycle estimate.
        """
        sig = pd.Series(sig).interpolate(limit_direction="both").ffill().bfill().values
        if len(sig) < 5 or np.all(np.isnan(sig)):
            return 0

        w = min(11, len(sig))
        if w % 2 == 0:
            w -= 1
        if w >= 5:
            sig = savgol_filter(sig, w, min(3, w - 2))

        sig_range = np.nanpercentile(sig, 95) - np.nanpercentile(sig, 5)
        if not np.isfinite(sig_range) or sig_range < 1e-6:
            return 0

        centered = sig - np.nanmedian(sig)
        min_distance = max(1, int(round(frame_rate * min_interval_sec)))
        prominence = max(sig_range * 0.08, np.nanstd(centered) * 0.20)
        peaks, _ = find_peaks(centered, distance=min_distance, prominence=prominence)
        troughs, _ = find_peaks(-centered, distance=min_distance, prominence=prominence)
        return int(max(len(peaks), len(troughs)))


    def build_bout_step_counts(df: pd.DataFrame,
                               bout_ids: np.ndarray,
                               frame_rate: float,
                               min_interval_sec: float) -> dict:
        step_counts = {}
        for b in np.unique(bout_ids[bout_ids > 0]):
            idx = np.where(bout_ids == b)[0]
            foot_steps = count_steps_in_signal(
                df["foot_x"].iloc[idx].values,
                frame_rate=frame_rate,
                min_interval_sec=min_interval_sec,
            )
            ankle_steps = count_steps_in_signal(
                df["ankle_x"].iloc[idx].values,
                frame_rate=frame_rate,
                min_interval_sec=min_interval_sec,
            )
            step_counts[b] = max(foot_steps, ankle_steps)
        return step_counts


    def filter_bouts_by_step_count(bout_ids: np.ndarray,
                                   step_counts: dict,
                                   min_steps: int):
        if min_steps <= 0:
            return bout_ids > 0, bout_ids

        keep = {b for b, n_steps in step_counts.items() if n_steps >= min_steps}
        filtered_ids = np.zeros(len(bout_ids), dtype=int)
        next_id = 0
        for b in np.unique(bout_ids[bout_ids > 0]):
            if b not in keep:
                continue
            next_id += 1
            filtered_ids[bout_ids == b] = next_id
        return filtered_ids > 0, filtered_ids


    def build_bout_metadata(df_raw: pd.DataFrame,
                            bout_ids: np.ndarray,
                            quality: dict,
                            step_counts: dict,
                            frame_rate: float) -> pd.DataFrame:
        rows = []
        total_frames = len(df_raw)
        total_duration_sec = total_frames / frame_rate if frame_rate > 0 else np.nan
        for b in np.unique(bout_ids[bout_ids > 0]):
            idx = np.where(bout_ids == b)[0]
            fr = df_raw["frame"].iloc[idx]
            n_frames = len(idx)
            duration_sec = n_frames / frame_rate if frame_rate > 0 else np.nan
            n_steps = int(step_counts.get(b, 0))
            step_frequency_hz = (
                n_steps / duration_sec
                if duration_sec and np.isfinite(duration_sec) and duration_sec > 0
                else np.nan
            )
            duration_total_fraction = (
                duration_sec / total_duration_sec
                if total_duration_sec and np.isfinite(total_duration_sec) and total_duration_sec > 0
                else np.nan
            )
            rows.append({
                "bout_id":                 b,
                "start_frame":             int(fr.iloc[0]),
                "end_frame":               int(fr.iloc[-1]),
                "n_frames":                n_frames,
                "duration_sec":            duration_sec,
                "duration_total_fraction": duration_total_fraction,
                "n_steps":                 n_steps,
                "step_frequency_hz":       step_frequency_hz,
                "step_quality":            quality.get(b, 0.0),
                "bout_type":               "stepping" if quality.get(b, 0) >= STEP_QUALITY_BAND_FRAC else "dragging/irregular",
            })
        return pd.DataFrame(rows)
    
    
    # ── 7. SMOOTHING ──────────────────────────────────────────────
    def smooth_bouts(df: pd.DataFrame, parts: list,
                     bout_ids: np.ndarray, window: int, poly: int) -> pd.DataFrame:
        df = df.copy()
        for b in np.unique(bout_ids[bout_ids > 0]):
            idx = np.where(bout_ids == b)[0]
            for p in parts:
                for coord in ["x", "y"]:
                    col = f"{p}_{coord}"
                    if col not in df.columns:
                        continue
                    seg = df[col].iloc[idx].values.copy()
                    seg_filled = pd.Series(seg).interpolate(limit_direction="both").values
                    w = window if window % 2 == 1 else window - 1
                    w = min(w, len(seg_filled))
                    if w % 2 == 0:
                        w -= 1
                    if w < poly + 2 or w < 3:
                        continue
                    df.iloc[idx, df.columns.get_loc(col)] = savgol_filter(seg_filled, w, poly)
        return df
    
    
    # ── 8. EXPORT ─────────────────────────────────────────────────
    def build_cleaned_df(df: pd.DataFrame, bout_mask: np.ndarray,
                         bout_ids: np.ndarray, quality: dict,
                         bout_meta: pd.DataFrame) -> pd.DataFrame:
        out = df[bout_mask].copy()
        out["bout_id"] = bout_ids[bout_mask]
        out["step_quality"] = out["bout_id"].map(quality)
        if not bout_meta.empty:
            meta_cols = [
                "bout_id",
                "duration_sec",
                "duration_total_fraction",
                "n_steps",
                "step_frequency_hz",
            ]
            out = out.merge(bout_meta[meta_cols], on="bout_id", how="left")
        return out
    
    
    # ─────────────────────── DIAGNOSTIC PLOTS ───────────────────────
    
    def _bout_colors(unique_bouts):
        cmap = plt.cm.tab20
        return {b: cmap(i / max(len(unique_bouts), 1)) for i, b in enumerate(unique_bouts)}
    
    
    def plot_likelihood_overview(df_raw: pd.DataFrame, parts: list, thresh: dict, out_dir: Path):
        fig, axes = plt.subplots(len(parts), 1, figsize=(14, 2.2 * len(parts)), sharex=True)
        colors = plt.cm.tab10.colors
        for ax, p, c in zip(axes, parts, colors):
            t = thresh.get(p, 0.6)
            ax.plot(df_raw["frame"], df_raw[f"{p}_likelihood"], color=c, lw=0.8)
            ax.axhline(t, color="red", ls="--", lw=1, label=f"thresh={t}")
            below = (df_raw[f"{p}_likelihood"] < t)
            ax.fill_between(df_raw["frame"], 0, df_raw[f"{p}_likelihood"],
                            where=below, alpha=0.3, color="red", label="rejected")
            ax.set_ylabel("likelihood (normalised)", fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_title(p, fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
        axes[-1].set_xlabel("frame")
        fig.suptitle("Stage 2 — Per-bodypart likelihood normalised (red = masked)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / f"{INPUT_CSV.stem}_01_likelihood_filtering.png", dpi=150)
        plt.close(fig)
    
    
    def plot_motion_score(df_raw: pd.DataFrame, smooth_score: pd.Series,
                          bout_ids: np.ndarray, motion_thresh: float, out_dir: Path):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax1.plot(df_raw["frame"], smooth_score.values, lw=0.9, color="navy")
        ax1.axhline(motion_thresh, color="red", ls="--", lw=1.2, label=f"thresh={motion_thresh}")
        ax1.set_ylabel("motion score")
        ax1.set_title("Stage 5 — Composite motion score (rolling median)")
        ax1.legend(fontsize=8)
    
        ubouts = np.unique(bout_ids[bout_ids > 0])
        bc = _bout_colors(ubouts)
        for b in ubouts:
            idx = np.where(bout_ids == b)[0]
            ax2.axvspan(df_raw["frame"].iloc[idx[0]], df_raw["frame"].iloc[idx[-1]],
                        alpha=0.35, color=bc[b], label=f"bout {b}")
        ax2.fill_between(df_raw["frame"], (bout_ids > 0).astype(int),
                         step="mid", alpha=0.2, color="steelblue")
        ax2.set_ylabel("in bout")
        ax2.set_xlabel("frame")
        ax2.set_title(f"Detected bouts ({len(ubouts)} total)")
        handles = [mpatches.Patch(color=bc[b], label=f"bout {b}") for b in ubouts]
        ax2.legend(handles=handles, fontsize=6, loc="upper right", ncol=5)
        plt.tight_layout()
        fig.savefig(out_dir / f"{INPUT_CSV.stem}_02_motion_bouts.png", dpi=150)
        plt.close(fig)
    
    
    def plot_xy_overview(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                         bout_ids: np.ndarray, parts: list, out_dir: Path):
        ubouts = np.unique(bout_ids[bout_ids > 0])
        bc = _bout_colors(ubouts)
    
        fig, axes = plt.subplots(len(parts), 2, figsize=(16, 2.8 * len(parts)), sharex=True)
        for row, p in enumerate(parts):
            ax_x = axes[row, 0]
            ax_y = axes[row, 1]
            # raw (grey)
            ax_x.plot(df_raw["frame"], df_raw[f"{p}_x"], color="lightgrey", lw=0.6, zorder=1)
            ax_y.plot(df_raw["frame"], df_raw[f"{p}_y"], color="lightgrey", lw=0.6, zorder=1)
            # cleaned bouts
            for b in ubouts:
                seg = df_clean[df_clean["bout_id"] == b]
                ax_x.plot(seg["frame"], seg[f"{p}_x"], color=bc[b], lw=1.0, zorder=2)
                ax_y.plot(seg["frame"], seg[f"{p}_y"], color=bc[b], lw=1.0, zorder=2)
            ax_x.set_title(f"{p}  x", fontsize=8)
            ax_y.set_title(f"{p}  y", fontsize=8)
            ax_x.set_ylabel("px", fontsize=7)
            ax_y.set_ylabel("px", fontsize=7)
            ax_y.invert_yaxis()
        for ax in axes[-1]:
            ax.set_xlabel("frame")
        patches = [mpatches.Patch(color=bc[b], label=f"bout {b}") for b in ubouts]
        fig.legend(handles=patches, loc="upper right", fontsize=6, ncol=6)
        fig.suptitle("Stage 7 — Raw (grey) vs Cleaned bouts (colour-coded)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / f"{INPUT_CSV.stem}_03_xy_raw_vs_clean.png", dpi=150)
        plt.close(fig)
    
    
    def plot_step_quality(quality: dict, bout_meta: pd.DataFrame, out_dir: Path):
        if not quality:
            return
        bouts = sorted(quality.keys())
        scores = [quality[b] for b in bouts]
        durations = [bout_meta.loc[bout_meta["bout_id"] == b, "n_frames"].values[0]
                     if b in bout_meta["bout_id"].values else 0 for b in bouts]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        bars = ax1.bar(bouts, scores, color=plt.cm.RdYlGn(np.array(scores)))
        ax1.set_xlabel("Bout ID")
        ax1.set_ylabel("Step quality score (spectral regularity)")
        ax1.set_title("Step quality per bout\n(higher = more rhythmic stepping)")
        ax1.set_xticks(bouts)
        ax1.axhline(STEP_QUALITY_BAND_FRAC, color="red", ls="--", lw=1,
                    label=f"stepping threshold ({STEP_QUALITY_BAND_FRAC})")
        ax1.legend(fontsize=8)
    
        sc = ax2.scatter(durations, scores, c=scores, cmap="RdYlGn", s=80, zorder=3)
        for b, d, s in zip(bouts, durations, scores):
            ax2.annotate(f"b{b}", (d, s), textcoords="offset points", xytext=(4, 4), fontsize=7)
        plt.colorbar(sc, ax=ax2, label="quality")
        ax2.set_xlabel("Bout duration (frames)")
        ax2.set_ylabel("Step quality")
        ax2.set_title("Duration vs Quality")
        plt.tight_layout()
        fig.savefig(out_dir / f"{INPUT_CSV.stem}_04_step_quality.png", dpi=150)
        plt.close(fig)
    
    
    def plot_sample_bouts(df_clean: pd.DataFrame, parts: list, quality: dict, out_dir: Path):
        """Plot ankle x + y traces within each bout (phase portrait view)."""
        ubouts = sorted(df_clean["bout_id"].unique())
        n = len(ubouts)
        if n == 0:
            return
        ncols = min(n, 4)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
        axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])
        cmap = plt.cm.tab20
    
        for i, b in enumerate(ubouts):
            ax = axes[i]
            seg = df_clean[df_clean["bout_id"] == b]
            t   = np.arange(len(seg))
            ax.plot(t, seg["ankle_x"].values, color="steelblue", lw=1.0, label="ankle x")
            ax.plot(t, seg["foot_x"].values,  color="coral",     lw=1.0, label="foot x")
            ax2 = ax.twinx()
            ax2.plot(t, seg["ankle_y"].values, color="steelblue", lw=0.8, ls="--", alpha=0.6)
            ax2.plot(t, seg["foot_y"].values,  color="coral",     lw=0.8, ls="--", alpha=0.6)
            ax2.invert_yaxis()
            ax2.set_ylabel("y (dashed, inv)", fontsize=6, color="grey")
            q = quality.get(b, 0)
            label_str = "stepping" if q >= STEP_QUALITY_BAND_FRAC else "dragging/irregular"
            ax.set_title(f"Bout {b} | {len(seg)} fr\nquality={q:.2f} → {label_str}", fontsize=8)
            ax.set_xlabel("frame within bout", fontsize=7)
            ax.set_ylabel("x (px)", fontsize=7)
            ax.legend(fontsize=6, loc="upper right")
        # hide unused
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Stage 6 — Individual bout traces (ankle + foot x=solid, y=dashed)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / f"{INPUT_CSV.stem}_05_individual_bouts.png", dpi=150)
        plt.close(fig)
    
    
    # ─────────────────────── SUMMARY ────────────────────────────────
    def print_summary(df_raw, df_clean, bout_ids, parts, quality, bout_meta):
        total = len(df_raw)
        kept  = (bout_ids > 0).sum()
        ubouts = np.unique(bout_ids[bout_ids > 0])
        print(f"\n{'='*60}")
        print(f"  GAIT CLEANING PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"  Total frames       : {total}")
        print(f"  Frames in bouts    : {kept}  ({100*kept/total:.1f}%)")
        print(f"  Frames excluded    : {total-kept}  ({100*(total-kept)/total:.1f}%)")
        print(f"  Valid bouts found  : {len(ubouts)}")
        print()
        print(f"  {'Bout':>5}  {'Start':>6}  {'End':>6}  {'Frames':>7}  {'Sec':>7}  {'Steps':>6}  {'Hz':>6}  {'%Rec':>6}  {'Quality':>9}  {'Type'}")
        print(f"  {'-'*95}")
        for b in ubouts:
            idx = np.where(bout_ids == b)[0]
            fr  = df_raw["frame"].iloc[idx]
            q   = quality.get(b, 0)
            meta = bout_meta.loc[bout_meta["bout_id"] == b].iloc[0] if not bout_meta.empty else {}
            duration_sec = meta.get("duration_sec", np.nan) if hasattr(meta, "get") else np.nan
            n_steps = meta.get("n_steps", 0) if hasattr(meta, "get") else 0
            step_frequency_hz = meta.get("step_frequency_hz", np.nan) if hasattr(meta, "get") else np.nan
            duration_pct = 100 * meta.get("duration_total_fraction", np.nan) if hasattr(meta, "get") else np.nan
            btype = "stepping" if q >= STEP_QUALITY_BAND_FRAC else "dragging/irregular"
            print(
                f"  {b:>5}  {fr.iloc[0]:>6}  {fr.iloc[-1]:>6}  {len(idx):>7}  "
                f"{duration_sec:>7.2f}  {n_steps:>6}  {step_frequency_hz:>6.2f}  "
                f"{duration_pct:>6.2f}  {q:>9.3f}  {btype}"
            )
        print()
        print(f"  NaN rates in cleaned output:")
        for p in parts:
            n_nan = df_clean[f"{p}_x"].isna().sum()
            pct   = 100 * n_nan / max(len(df_clean), 1)
            bar   = "█" * int(pct / 5)
            print(f"    {p:8s}: {n_nan:4d}/{len(df_clean):4d}  {pct:5.1f}%  {bar}")
        print('='*60)
    
    
    # ─────────────────────────── MAIN ────────────────────────────────
    if __name__ == "__main__":
        print(f"{'─'*60}")
        print(f"  Gait Cleaning Pipeline  |  DLC Treadmill Mouse")
        print(f"{'─'*60}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    
        print(f"\n[1/8] Parsing DLC CSV…")
        df_raw, parts = parse_dlc_csv(INPUT_CSV)
        print(f"      {len(df_raw)} frames  |  bodyparts: {parts}")
        likelihood_cols = [f"{p}_likelihood" for p in parts if f"{p}_likelihood" in df_raw.columns]
        mean_raw_likelihood = df_raw[likelihood_cols].mean().mean() if likelihood_cols else np.nan
        high_confidence_tracking = (
            np.isfinite(mean_raw_likelihood)
            and mean_raw_likelihood >= HIGH_CONFIDENCE_MEAN_LIKELIHOOD
        )
        if high_confidence_tracking:
            detect_roll_window = MOTION_ROLL_WINDOW
            detect_off_thresh = MOTION_OFF_THRESH
            detect_dilation = MOTION_DILATION_FRAMES
            detect_max_gap = MAX_INACTIVE_GAP_FRAMES
            detect_min_frames = MIN_BOUT_FRAMES
            detect_min_steps = MIN_STEPS_PER_BOUT
            detect_mode = "high-confidence sustained stepping"
        else:
            detect_roll_window = LEGACY_MOTION_ROLL_WINDOW
            detect_off_thresh = LEGACY_MOTION_OFF_THRESH
            detect_dilation = LEGACY_MOTION_DILATION_FRAMES
            detect_max_gap = LEGACY_MAX_INACTIVE_GAP_FRAMES
            detect_min_frames = LEGACY_MIN_BOUT_FRAMES
            detect_min_steps = LEGACY_MIN_STEPS_PER_BOUT
            detect_mode = "legacy low-confidence"
        print(f"      mean raw likelihood={mean_raw_likelihood:.3f}  |  mode: {detect_mode}")
    
        print(f"\n[2/8] Likelihood filtering…")
        df = apply_likelihood_filter(df_raw, parts, LIKELIHOOD_THRESH)
        for p in parts:
            n_masked = df[f"{p}_x"].isna().sum() - df_raw[f"{p}_x"].isna().sum()
            print(f"      {p:8s}: {n_masked} frames masked")
    
        print(f"\n[2b] Normalising likelihoods per body part (min-max)…")
        df = normalize_likelihoods(df, parts)
        for p in parts:
            col = df[f"{p}_likelihood"]
            print(f"      {p:8s}: min={col.min():.3f}  max={col.max():.3f}  mean={col.mean():.3f}")
    
        print(f"\n[3/8] Velocity spike rejection (thresh={VELOCITY_SPIKE_THRESH} px/frame)…")
        df = reject_velocity_spikes(df, parts, VELOCITY_SPIKE_THRESH)
    
        print(f"\n[4/8] Interpolating short gaps (max {MAX_INTERP_GAP} frames)…")
        df = interpolate_gaps(df, parts, MAX_INTERP_GAP)
    
        print(f"\n[5/8] Detecting motion bouts…")
        bout_mask, bout_ids, smooth_score = detect_bouts(
            df,
            motion_thresh=MOTION_THRESH,
            motion_off_thresh=detect_off_thresh,
            roll_window=detect_roll_window,
            dilation=detect_dilation,
            max_inactive_gap=detect_max_gap,
            min_frames=detect_min_frames,
        )
        preliminary_steps = build_bout_step_counts(
            df,
            bout_ids,
            frame_rate=FRAME_RATE,
            min_interval_sec=MIN_STEP_INTERVAL_SEC,
        )
        bout_mask, bout_ids = filter_bouts_by_step_count(
            bout_ids,
            preliminary_steps,
            min_steps=detect_min_steps,
        )
        ubouts = np.unique(bout_ids[bout_ids > 0])
        step_counts = build_bout_step_counts(
            df,
            bout_ids,
            frame_rate=FRAME_RATE,
            min_interval_sec=MIN_STEP_INTERVAL_SEC,
        )
        print(f"      Found {len(ubouts)} valid bouts  ({bout_mask.sum()} frames total)")
        print(f"      Merged inactive gaps <= {detect_max_gap} frames")
        if detect_min_steps > 0:
            print(f"      Removed candidate bouts with < {detect_min_steps} detected steps")
        else:
            print(f"      Step count filtering disabled; step counts are reported as features")
    
        print(f"\n[6/8] Scoring bout periodicity (step quality)…")
        quality = score_bout_periodicity(df, bout_ids, STEP_QUALITY_BAND_FRAC)
        for b, q in quality.items():
            btype = "stepping" if q >= STEP_QUALITY_BAND_FRAC else "dragging/irregular"
            print(f"      Bout {b:2d}: steps={step_counts.get(b, 0):2d}  quality={q:.3f}  → {btype}")
    
        print(f"\n[7/8] Smoothing within bouts (SG w={SG_WINDOW}, poly={SG_POLY})…")
        df_smoothed = smooth_bouts(df, parts, bout_ids, SG_WINDOW, SG_POLY)
    
        print(f"\n[8/8] Exporting outputs to {OUT_DIR}/")
        bout_meta = build_bout_metadata(df_raw, bout_ids, quality, step_counts, FRAME_RATE)
        df_clean = build_cleaned_df(df_smoothed, bout_mask, bout_ids, quality, bout_meta)
        df_clean.to_csv(OUT_DIR / f"{INPUT_CSV.stem}_cleaned_gait_data.csv", index=False)
        bout_meta.to_csv(OUT_DIR / f"{INPUT_CSV.stem}_bout_metadata.csv", index=False)
    
        print("      Generating diagnostic plots…")
        # Pass df (post-normalisation) so the plot reflects normalised likelihoods
        plot_likelihood_overview(df, parts, LIKELIHOOD_THRESH, OUT_DIR)
        plot_motion_score(df_raw, smooth_score, bout_ids, MOTION_THRESH, OUT_DIR)
        plot_xy_overview(df_raw, df_clean, bout_ids, parts, OUT_DIR)
        plot_step_quality(quality, bout_meta, OUT_DIR)
        plot_sample_bouts(df_clean, parts, quality, OUT_DIR)
    
        print_summary(df_raw, df_clean, bout_ids, parts, quality, bout_meta)
        print(f"\n  Output files:")
        print(f"    {INPUT_CSV.stem}_cleaned_gait_data.csv  — per-frame cleaned coords + bout features")
        print(f"    {INPUT_CSV.stem}_bout_metadata.csv      — bout-level summary")
        print(f"    {INPUT_CSV.stem}_01_likelihood_filtering.png")
        print(f"    {INPUT_CSV.stem}_02_motion_bouts.png")
        print(f"    {INPUT_CSV.stem}_03_xy_raw_vs_clean.png")
        print(f"    {INPUT_CSV.stem}_04_step_quality.png")
        print(f"    {INPUT_CSV.stem}_05_individual_bouts.png")
        print()
