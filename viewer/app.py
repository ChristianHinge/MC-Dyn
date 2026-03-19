"""MC-Dyn Interactive Viewer — TAC visualization and on-the-fly kinetic analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="MC-Dyn Viewer", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    header[data-testid="stHeader"] {
        backdrop-filter: blur(10px);
        height: 2.5rem;
        position: fixed;
        top: 0;
        z-index: 999;
    }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


@st.cache_data
def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir)
    tacs = pd.read_csv(d / "tacs.csv")
    studies = pd.read_csv(d / "studies.csv")
    # Merge demographics onto TACs
    tacs = tacs.merge(studies[["case_id", "age", "sex", "weight_kg", "injected_dose_mbq"]], on="case_id", how="left")
    # SUV = mean_value / (dose_Bq / weight_g) = mean_value * weight_kg / (dose_MBq * 1e3)
    if (tacs["injected_dose_mbq"] > 0).any() and (tacs["weight_kg"] > 0).any():
        tacs["suv"] = tacs["mean_value"] * tacs["weight_kg"] / (tacs["injected_dose_mbq"] * 1e3)
    else:
        tacs["suv"] = np.nan
    tacs["time_min"] = tacs["time_mid_s"] / 60.0
    return tacs, studies


# ---------------------------------------------------------------------------
# Sidebar — data path
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<h1 style="margin-top:-1rem;margin-bottom:1rem;">MC-Dyn</h1>', unsafe_allow_html=True)
    data_dir = st.text_input("Output directory", value="out_data", help="Path to mc-dyn output directory containing tacs.csv and studies.csv")

try:
    tacs_all, studies = load_data(data_dir)
except FileNotFoundError as e:
    st.error(f"Could not load data from **{data_dir}**: {e}")
    st.stop()

# Derived constants
all_cases = sorted(tacs_all["case_id"].unique())
all_tasks = sorted(tacs_all["task"].unique())

# Organ lists per task
organ_tacs = sorted(tacs_all[tacs_all["task"].str.startswith("moosez")]["organ"].unique())
input_functions = sorted(tacs_all[tacs_all["task"] == "nifti_dynamic"]["organ"].unique())

max_frame_idx = int(tacs_all["frame_idx"].max())
max_time_min = float(tacs_all["time_min"].max())

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Time Activity Curves", "Static Uptake", "Patlak Kᵢ"])

# ===========================================================================
# TAB 1 — Time Activity Curves
# ===========================================================================

with tab1:
    with st.sidebar:
        st.markdown("---")
        st.markdown("**TAC settings**")

        task_sel = st.selectbox(
            "Segmentation task",
            options=all_tasks,
            key="tac_task",
        )
        organs_for_task = sorted(tacs_all[tacs_all["task"] == task_sel]["organ"].unique())
        default_organs = organs_for_task[:3] if len(organs_for_task) >= 3 else organs_for_task
        selected_organs_tac = st.multiselect(
            "Organs",
            options=organs_for_task,
            default=default_organs,
            key="tac_organs",
        )

        color_by_tac = st.selectbox(
            "Color by",
            ["Organ", "Case", "Sex"],
            key="tac_color",
        )

        y_axis_tac = st.selectbox(
            "Y-axis",
            ["Mean value [Bq/mL]", "SUV"] if not tacs_all["suv"].isna().all() else ["Mean value [Bq/mL]"],
            key="tac_y",
        )

        uncertainty_tac = st.selectbox(
            "Uncertainty",
            ["None", "Std dev", "95% CI", "95% PI"],
            index=2,
            key="tac_uncertainty",
        )

        x_axis_tac = st.selectbox("X-axis", ["Time [min]", "Frame index"], key="tac_x")

        time_range_tac = st.slider(
            "Time range [min]",
            min_value=0.0,
            max_value=max_time_min,
            value=(0.0, max_time_min),
            step=0.5,
            key="tac_time_range",
        )

    # --- Build plot ---
    df = tacs_all[
        (tacs_all["task"] == task_sel) &
        (tacs_all["organ"].isin(selected_organs_tac)) &
        (tacs_all["time_min"] >= time_range_tac[0]) &
        (tacs_all["time_min"] <= time_range_tac[1])
    ].copy()

    y_col = "suv" if y_axis_tac == "SUV" else "mean_value"
    x_col = "time_min" if x_axis_tac == "Time [min]" else "frame_idx"

    color_col_map = {"Organ": "organ", "Case": "case_id", "Sex": "sex"}
    color_col = color_col_map[color_by_tac]

    if df.empty:
        st.warning("No data for selected filters.")
    else:
        fig = go.Figure()
        color_groups = sorted(df[color_col].dropna().unique())

        for i, grp_val in enumerate(color_groups):
            gdf = df[df[color_col] == grp_val]
            summary = (
                gdf.groupby(x_col)[y_col]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values(x_col)
            )

            if uncertainty_tac == "95% CI":
                unc = 1.96 * summary["std"] / np.sqrt(summary["count"])
            elif uncertainty_tac == "95% PI":
                unc = 1.96 * summary["std"]
            elif uncertainty_tac == "Std dev":
                unc = summary["std"]
            else:
                unc = pd.Series(0.0, index=summary.index)
            unc = unc.fillna(0)

            c = COLORS[i % len(COLORS)]
            rgb = tuple(int(c.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
            fill_c = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)"

            fig.add_trace(go.Scatter(
                x=summary[x_col], y=summary["mean"],
                name=str(grp_val), mode="lines+markers",
                line=dict(color=c, width=2), marker=dict(size=4),
                hovertemplate=f"<b>{grp_val}</b><br>{x_axis_tac}: %{{x:.1f}}<br>Value: %{{y:.3f}}<extra></extra>",
            ))

            if uncertainty_tac != "None":
                fig.add_trace(go.Scatter(
                    x=summary[x_col], y=summary["mean"] + unc,
                    mode="lines", line_color="rgba(0,0,0,0)", showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=summary[x_col], y=summary["mean"] - unc,
                    mode="lines", line_color="rgba(0,0,0,0)", fill="tonexty",
                    fillcolor=fill_c, showlegend=False, hoverinfo="skip",
                ))

        fig.update_layout(
            xaxis_title=x_axis_tac,
            yaxis_title=y_axis_tac,
            hovermode="x unified",
            height=500,
            template="plotly_white",
            legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98),
            margin=dict(l=60, r=20, t=20, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# TAB 2 — Static Uptake (mean over selected frames)
# ===========================================================================

with tab2:
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Static uptake settings**")

        task_static = st.selectbox("Segmentation task", options=[t for t in all_tasks if t != "nifti_dynamic"], key="static_task")
        organs_static_all = sorted(tacs_all[tacs_all["task"] == task_static]["organ"].unique())
        selected_organs_static = st.multiselect(
            "Organs",
            options=organs_static_all,
            default=organs_static_all[:8],
            key="static_organs",
        )

        frame_range_static = st.slider(
            "Frame range (inclusive)",
            min_value=0,
            max_value=max_frame_idx,
            value=(max(0, max_frame_idx - 5), max_frame_idx),
            key="static_frames",
        )
        st.caption(
            f"Frames {frame_range_static[0]}–{frame_range_static[1]}  "
            f"({tacs_all[tacs_all['frame_idx'] == frame_range_static[0]]['time_min'].median():.1f}–"
            f"{tacs_all[tacs_all['frame_idx'] == frame_range_static[1]]['time_min'].median():.1f} min)"
        )

        color_by_static = st.selectbox("Color by", ["None", "Case", "Sex"], key="static_color")

    df_s = tacs_all[
        (tacs_all["task"] == task_static) &
        (tacs_all["organ"].isin(selected_organs_static)) &
        (tacs_all["frame_idx"] >= frame_range_static[0]) &
        (tacs_all["frame_idx"] <= frame_range_static[1])
    ].groupby(["case_id", "organ", "sex"])["mean_value"].mean().reset_index()
    df_s.rename(columns={"mean_value": "static_mean"}, inplace=True)

    if df_s.empty:
        st.warning("No data for selected filters.")
    else:
        color_map = {"None": None, "Case": "case_id", "Sex": "sex"}
        color_col_s = color_map[color_by_static]

        fig_s = px.box(
            df_s, x="organ", y="static_mean", color=color_col_s,
            template="plotly_white",
        )
        fig_s.update_traces(
            boxmean=True,
            hovertemplate="<b>%{x}</b><br>Value: %{y:.1f} Bq/mL<extra></extra>",
        )
        fig_s.update_layout(
            xaxis_title="Organ", yaxis_title="Mean uptake [Bq/mL]",
            height=500, margin=dict(l=60, r=20, t=20, b=100),
            xaxis_tickangle=45,
        )
        st.plotly_chart(fig_s, use_container_width=True)

        # Summary table
        summary_s = (
            df_s.groupby("organ")["static_mean"]
            .agg(n="count", mean="mean", std="std", median="median")
            .round(1)
        )
        summary_s["CV [%]"] = (summary_s["std"] / summary_s["mean"].abs() * 100).round(1)
        st.dataframe(summary_s, use_container_width=True)

# ===========================================================================
# TAB 3 — Patlak Ki (on-the-fly)
# ===========================================================================

with tab3:
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Patlak settings**")

        if not input_functions:
            st.warning("No aorta input function TACs found (task=nifti_dynamic).")
        else:
            input_fn_sel = st.selectbox(
                "Input function",
                options=input_functions,
                index=input_functions.index("aorta_if_descending_bottom_1ml") if "aorta_if_descending_bottom_1ml" in input_functions else 0,
                key="patlak_if",
            )

        task_patlak = st.selectbox("Segmentation task", options=[t for t in all_tasks if t != "nifti_dynamic"], key="patlak_task")
        organs_patlak_all = sorted(tacs_all[tacs_all["task"] == task_patlak]["organ"].unique())
        selected_organs_patlak = st.multiselect(
            "Organs",
            options=organs_patlak_all,
            default=organs_patlak_all[:5],
            key="patlak_organs",
        )

        patlak_frame_range = st.slider(
            "Frames for linear fit",
            min_value=0,
            max_value=max_frame_idx,
            value=(max(0, max_frame_idx - 10), max_frame_idx),
            key="patlak_frames",
        )
        st.caption(
            f"Frames {patlak_frame_range[0]}–{patlak_frame_range[1]}  "
            f"({tacs_all[tacs_all['frame_idx'] == patlak_frame_range[0]]['time_min'].median():.1f}–"
            f"{tacs_all[tacs_all['frame_idx'] == patlak_frame_range[1]]['time_min'].median():.1f} min)"
        )

        color_by_patlak = st.selectbox("Color by", ["Organ", "Case", "Sex"], key="patlak_color")

    if not input_functions:
        st.warning("No input function TACs found. Run mc-dyn with `--model clin_ct_cardiac` to enable Patlak analysis.")
        st.stop()

    # --- Compute Patlak Ki per (case, organ) ---
    @st.cache_data
    def compute_patlak(
        tacs_csv_hash: str,
        input_fn: str,
        task: str,
        organs: tuple,
        fit_frames: tuple,
    ) -> pd.DataFrame:
        """Compute Patlak Ki for all cases. Returns rows: case_id, organ, ki, intercept."""
        results = []
        cases = tacs_all["case_id"].unique()
        for case in cases:
            # Input function
            cp_df = (
                tacs_all[(tacs_all["case_id"] == case) & (tacs_all["organ"] == input_fn)]
                .sort_values("frame_idx")
                .reset_index(drop=True)
            )
            if cp_df.empty:
                continue

            cp = cp_df["mean_value"].values
            dt = (cp_df["time_end_s"] - cp_df["time_start_s"]).values  # frame durations
            t_mid = cp_df["time_mid_s"].values

            # Cumulative integral of Cp up to each frame midpoint (trapezoidal)
            cum_integral = np.cumsum(cp * dt)

            # Normalised time: int Cp / Cp(t)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_all = np.where(cp > 0, cum_integral / cp, np.nan)

            # Organ TACs
            for organ in organs:
                ct_df = (
                    tacs_all[(tacs_all["case_id"] == case) & (tacs_all["task"] == task) & (tacs_all["organ"] == organ)]
                    .sort_values("frame_idx")
                    .reset_index(drop=True)
                )
                if ct_df.empty:
                    continue

                ct = ct_df["mean_value"].values
                sex = ct_df["sex"].iloc[0] if "sex" in ct_df.columns else None

                # Fit range
                mask = (ct_df["frame_idx"] >= fit_frames[0]) & (ct_df["frame_idx"] <= fit_frames[1])
                x_fit = x_all[mask]
                y_fit = np.where(cp[mask] > 0, ct[mask] / cp[mask], np.nan)

                valid = ~(np.isnan(x_fit) | np.isnan(y_fit))
                if valid.sum() < 2:
                    continue

                slope, intercept = np.polyfit(x_fit[valid], y_fit[valid], 1)
                results.append({
                    "case_id": case,
                    "organ": organ,
                    "sex": sex,
                    "ki": slope,        # min^-1 if time in minutes (using seconds here → s^-1)
                    "intercept": intercept,
                })

        return pd.DataFrame(results)

    patlak_df = compute_patlak(
        tacs_csv_hash=data_dir,
        input_fn=input_fn_sel,
        task=task_patlak,
        organs=tuple(sorted(selected_organs_patlak)),
        fit_frames=patlak_frame_range,
    )

    if patlak_df.empty:
        st.warning("Could not compute Patlak Ki. Check input function and organ selections.")
    else:
        # Convert Ki to 10^-3 min^-1 (from s^-1: multiply by 60*1000)
        patlak_df["ki_1000"] = patlak_df["ki"] * 60 * 1000

        color_map_p = {"Organ": "organ", "Case": "case_id", "Sex": "sex"}
        color_col_p = color_map_p[color_by_patlak]

        fig_p = px.box(
            patlak_df, x="organ", y="ki_1000", color=color_col_p,
            template="plotly_white",
        )
        fig_p.update_traces(
            boxmean=True,
            hovertemplate="<b>%{x}</b><br>Kᵢ: %{y:.3f} ×10⁻³ min⁻¹<extra></extra>",
        )
        fig_p.update_layout(
            xaxis_title="Organ",
            yaxis_title="Kᵢ [×10⁻³ min⁻¹]",
            height=500,
            margin=dict(l=60, r=20, t=20, b=100),
            xaxis_tickangle=45,
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Summary table
        summary_p = (
            patlak_df.groupby("organ")["ki_1000"]
            .agg(n="count", mean="mean", std="std", median="median")
            .round(4)
        )
        summary_p["CV [%]"] = (summary_p["std"] / summary_p["mean"].abs() * 100).round(1)
        st.dataframe(summary_p, use_container_width=True)
