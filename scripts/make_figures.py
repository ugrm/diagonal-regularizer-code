#!/usr/bin/env python
"""Draw every figure of the paper from the data in data/ into figures/.

Usage:
    python scripts/make_figures.py                      # all figures
    python scripts/make_figures.py --only fig02 fig04   # a subset (match by prefix)
    python scripts/make_figures.py --compare DIR        # also compare with reference PDFs in DIR
                                                        # (needs pdftoppm; renders both at 100 dpi)
"""
import argparse
import importlib
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")

# paper figure file(s) -> generator module in src/visualization
FIGURES = [
    (["fig04_learned_spectra"], "fig_learned_spectra"),          # Fig. 1
    (["fig02_flat_landscape"], "fig_flat_landscape"),            # Fig. 2
    (["fig13_heat_2d_sweep"], "fig_heat_2d_sweep"),              # Fig. 3
    (["fig_delta_vs_Eop"], "fig_delta_vs_Eop"),
    (["fig01_noise_profile"], "fig_noise_profile"),
    (["fig15_berry_qq_stratified"], "fig_berry_qq"),
    (["fig23_size_ablation"], "fig_size_ablation"),
    (["fig16_cost_distribution"], "fig_cost_distribution"),
    (["fig07_per_room_diagnostics"], "fig_per_room_diagnostics"),
    (["fig_m3_training"], "fig_m3_training"),
    (["fig_geometric_predictor"], "fig_geometric_predictor"),
    (["fig24_sweep_heatmap"], "fig_sweep_heatmap"),
    (["figS2_capacity_check"], "fig_capacity_check"),
    (["figS_lir_ablation"], "fig_lir_ablation"),
    (["fig10_heat_mode_survival", "fig11_heat_amplitude_spectrum"], "fig_heat_diagnostics"),
    (["fig17_heat_fit_residuals"], "fig_heat_fit_residuals"),
    (["fig14_c_vs_t_tracking"], "fig_c_vs_t_tracking"),
    (["fig03_p_sweep_curves"], "fig_p_sweep_curves"),
    (["fig_berry_weyl"], "fig_berry_weyl"),
]


def render(pdf, out_png):
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-singlefile", pdf, out_png[:-4]], check=True,
                   capture_output=True)


def compare(name, out_dir, ref_dir):
    import numpy as np
    from PIL import Image
    a, b = os.path.join(out_dir, name + ".pdf"), os.path.join(ref_dir, name + ".pdf")
    if not os.path.exists(b):
        return "no reference"
    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "a.png"), os.path.join(td, "b.png")
        render(a, pa); render(b, pb)
        x, y = np.asarray(Image.open(pa).convert("L")), np.asarray(Image.open(pb).convert("L"))
        if x.shape != y.shape:
            return f"size differs {x.shape} vs {y.shape}"
        n = int((np.abs(x.astype(int) - y.astype(int)) > 0).sum())
        return "identical" if n == 0 else f"{n} pixels differ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--output-dir", default="figures/")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--compare", metavar="DIR", help="folder with reference PDFs of the same names")
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    failed = []
    for names, mod in FIGURES:
        if a.only and not any(n.startswith(o) for n in names for o in a.only):
            continue
        try:
            importlib.import_module(f"src.visualization.{mod}").generate(data_dir=a.data_dir, output_dir=a.output_dir)
            status = "ok"
            if a.compare:
                status = "; ".join(f"{n}: {compare(n, a.output_dir, a.compare)}" for n in names)
        except Exception as e:
            status = f"FAILED: {e!r}"; failed.append(mod)
        print(f"{', '.join(names):<55} {status}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
