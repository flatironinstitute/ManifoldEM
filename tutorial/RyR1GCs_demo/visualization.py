"""Plotting helpers for the RyR1GCs ManifoldEM tutorial notebook.

Each `VizSession.show_*` method renders either a live ipywidgets/Plotly
viewer (`interactive=True`) or a static matplotlib snapshot that GitHub's
notebook viewer can display (`interactive=False`).
"""
from __future__ import annotations

import base64
import importlib
import pickle
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Patch
from scipy import stats


def _load_toml(toml_path: Path) -> dict:
    try:
        toml_parser = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        toml_parser = importlib.import_module("tomli")
    with toml_path.open("rb") as f:
        return toml_parser.load(f)


def load_session(toml_file_name: Optional[str] = None, cwd: Optional[Path] = None) -> "VizSession":
    """Locate the project TOML, load pd_data.pkl, and return a VizSession."""
    cwd = Path(cwd) if cwd else Path.cwd()

    if toml_file_name is None:
        candidates = sorted(cwd.glob("*.toml"))
        if not candidates:
            raise FileNotFoundError(f"No .toml file found in {cwd}.")
        toml_file_name = candidates[0].name
        print(f"Auto-detected TOML: {toml_file_name}")

    toml_file_name = toml_file_name.strip()
    if not toml_file_name.endswith(".toml"):
        toml_file_name += ".toml"

    toml_path = cwd / toml_file_name
    print(f"Using TOML file: {toml_path}")

    toml_config = _load_toml(toml_path)
    params_section = toml_config.get("params", toml_config)
    project_name = params_section.get("project_name", "")
    print(f"project_name: {project_name}")

    output_dir = cwd / "output" / project_name
    pkl_path = output_dir / "pd_data.pkl"
    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        pd_data = pickle.load(f)

    print(f"\nLoaded keys: {list(pd_data.keys())[:6]}…")
    print(f"  bin_centers       : {pd_data['bin_centers'].shape}")
    print(f"  occupancy_full    : {pd_data['occupancy_full'].shape}")
    print(f"  pos_thresholded   : {pd_data['pos_thresholded'].shape}")
    print(f"  thres_low / high  : {pd_data['thres_low']} / {pd_data['thres_high']}")

    return VizSession(pd_data=pd_data, output_dir=output_dir, project_name=project_name)


class VizSession:
    """Holds tutorial state shared across notebook cells.

    Cross-cell interactivity (sphere click → updates PrD slider, CC voting
    recolours sphere diamonds) requires sharing widget references; this
    object owns that state so the notebook cells stay one-liners.
    """

    def __init__(self, pd_data: dict, output_dir: Path, project_name: str):
        self.pd_data = pd_data
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.topos_dir = self.output_dir / "topos"

        self._cc_data: Optional[dict] = None
        self._traj_data: Optional[dict] = None

        # populated when interactive viewers are displayed
        self._sphere_fig = None
        self._prd_slider = None
        self._on_sphere_click = None

    # ── lazy data loaders ───────────────────────────────────────────────
    @property
    def cc_data(self) -> dict:
        if self._cc_data is None:
            with open(self.output_dir / "CC" / "CC_file.pkl", "rb") as f:
                self._cc_data = pickle.load(f)
        return self._cc_data

    @property
    def traj_data(self) -> dict:
        if self._traj_data is None:
            traj_dir = self.output_dir / "traj"
            with open(traj_dir / "traj_name1.pkl", "rb") as f:
                main = pickle.load(f)
            with open(traj_dir / "traj_name1_vars.pkl", "rb") as f:
                vars_ = pickle.load(f)
            self._traj_data = {"main": main, "vars": vars_}
        return self._traj_data

    # ─────────────────────────────────────────────────────────────────────
    # S2 sphere
    # ─────────────────────────────────────────────────────────────────────
    def show_s2_sphere(self, interactive: bool = True):
        if interactive:
            self._show_s2_sphere_interactive()
        else:
            self._show_s2_sphere_static()

    def _build_sphere_traces(self):
        pd = self.pd_data
        bin_centers = pd["bin_centers"]
        occupancy = pd["occupancy_full"]
        pos_thres = pd["pos_thresholded"]
        phi_thres = pd["phi_thresholded"]
        theta_thres = pd["theta_thresholded"]
        pos_full = pd["pos_full"]

        rng = np.random.default_rng(42)
        n_show = min(5000, pos_full.shape[1])
        idx = rng.choice(pos_full.shape[1], n_show, replace=False)
        pf = pos_full[:, idx]
        kde = stats.gaussian_kde(pf)
        density = kde(pf)
        density /= density.max()

        u, v = np.linspace(0, 2 * np.pi, 60), np.linspace(0, np.pi, 30)
        sphere = go.Surface(
            x=np.outer(np.cos(u), np.sin(v)),
            y=np.outer(np.sin(u), np.sin(v)),
            z=np.outer(np.ones_like(u), np.cos(v)),
            colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
            opacity=0.10, showscale=False, hoverinfo="skip", name="Unit sphere",
        )
        bins_trace = go.Scatter3d(
            x=bin_centers[0], y=bin_centers[1], z=bin_centers[2],
            mode="markers",
            marker=dict(
                size=4, color=occupancy, colorscale="Viridis",
                colorbar=dict(title="Occupancy<br>(particles/bin)", thickness=16, x=1.02),
                opacity=0.80, line=dict(width=0),
            ),
            text=[f"bin {i}<br>occ = {o}" for i, o in enumerate(occupancy)],
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
            name="Bin centres",
        )
        particles_trace = go.Scatter3d(
            x=pf[0], y=pf[1], z=pf[2], mode="markers",
            marker=dict(size=2, color=density, colorscale="Hot", opacity=0.35, line=dict(width=0)),
            hovertemplate="(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
            name=f"Particles (n={n_show}, KDE density)",
            visible="legendonly",
        )
        thres_trace = go.Scatter3d(
            x=pos_thres[0], y=pos_thres[1], z=pos_thres[2], mode="markers",
            marker=dict(size=8, color="red", opacity=0.95, symbol="diamond",
                        line=dict(color="darkred", width=1)),
            text=[f"PD {i+1}<br>φ={phi_thres[i]:.1f}°  θ={theta_thres[i]:.1f}°"
                  for i in range(len(phi_thres))],
            hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
            name=f"Active PDs (n={pos_thres.shape[1]})  ← click for class avg",
        )
        return sphere, bins_trace, thres_trace, particles_trace

    def _show_s2_sphere_interactive(self):
        import ipywidgets as widgets
        from IPython.display import display

        pd = self.pd_data
        sphere, bins_trace, thres_trace, particles_trace = self._build_sphere_traces()

        fig = go.FigureWidget(data=[sphere, bins_trace, thres_trace, particles_trace])
        fig.update_layout(
            title=dict(text="S2 Orientation Distribution — " + self.project_name, x=0.5),
            scene=dict(
                xaxis=dict(title="x", showgrid=False, zeroline=False),
                yaxis=dict(title="y", showgrid=False, zeroline=False),
                zaxis=dict(title="z", showgrid=False, zeroline=False),
                aspectmode="cube", bgcolor="rgb(240,240,248)",
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
            margin=dict(l=0, r=0, b=0, t=50), height=680,
        )

        status_label = widgets.Label(
            value="  ▶  Click a red diamond to view its 2D class average.",
            style={"font_size": "13px"},
        )
        avg_out = widgets.Output()

        phi_thres = pd["phi_thresholded"]
        theta_thres = pd["theta_thresholded"]
        occupancy = pd["occupancy_full"]
        thres_ids = list(pd["thres_ids"])

        def on_click(trace, points, selector):
            if not points.point_inds:
                return
            pd_idx = points.point_inds[0]
            prd_num = pd_idx + 1
            img_path = self.topos_dir / f"PrD_{prd_num}" / "class_avg.png"
            status_label.value = (
                f"  PrD {prd_num}  |  φ={phi_thres[pd_idx]:.1f}°  "
                f"θ={theta_thres[pd_idx]:.1f}°  |  "
                f"occ = {occupancy[thres_ids[pd_idx]]}"
            )
            with avg_out:
                avg_out.clear_output(wait=True)
                if not img_path.exists():
                    print(f"File not found: {img_path}")
                    return
                fig2, axes = plt.subplots(1, 2, figsize=(9, 4.5))
                img = mpimg.imread(str(img_path))
                axes[0].imshow(img, cmap="gray" if img.ndim == 2 else None)
                axes[0].set_title(
                    f"2D Class Average  —  PrD {prd_num}\n"
                    f"φ={phi_thres[pd_idx]:.1f}°  θ={theta_thres[pd_idx]:.1f}°  "
                    f"occ={occupancy[thres_ids[pd_idx]]}",
                    fontsize=11,
                )
                axes[0].axis("off")
                occ_active = [occupancy[thres_ids[i]] for i in range(len(thres_ids))]
                axes[1].bar(range(1, len(thres_ids) + 1), occ_active, color="steelblue", edgecolor="none")
                axes[1].bar(prd_num, occ_active[pd_idx], color="crimson", label=f"PrD {prd_num}")
                axes[1].axhline(pd["thres_low"], color="red", lw=1, ls="--", label="low thresh")
                axes[1].axhline(pd["thres_high"], color="green", lw=1, ls="--", label="high thresh")
                axes[1].set_xlabel("PrD number")
                axes[1].set_ylabel("Occupancy")
                axes[1].set_title("Occupancy of active PDs", fontsize=11)
                axes[1].legend(fontsize=9)
                plt.tight_layout()
                plt.show()
            # if the PrD slider has been created, sync it
            if self._prd_slider is not None:
                self._prd_slider.value = prd_num

        fig.data[2].on_click(on_click)
        self._sphere_fig = fig
        self._on_sphere_click = on_click

        display(widgets.VBox([fig, status_label, avg_out]))
        print(
            f"Active PDs: {pd['pos_thresholded'].shape[1]}  |  "
            f"Total bins: {len(pd['occupancy_full'])}  |  "
            f"Total particles: {pd['pos_full'].shape[1]}"
        )

    def _show_s2_sphere_static(self):
        """Render the Plotly sphere as a static PNG (via kaleido) so GitHub displays it."""
        pd = self.pd_data
        sphere, bins_trace, thres_trace, particles_trace = self._build_sphere_traces()
        fig = go.Figure(data=[sphere, bins_trace, thres_trace, particles_trace])
        fig.update_layout(
            title=dict(text=f"S2 Orientation Distribution — {self.project_name}", x=0.5),
            scene=dict(
                xaxis=dict(title="x", showgrid=False, zeroline=False),
                yaxis=dict(title="y", showgrid=False, zeroline=False),
                zaxis=dict(title="z", showgrid=False, zeroline=False),
                aspectmode="cube", bgcolor="rgb(240,240,248)",
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
            margin=dict(l=0, r=0, b=0, t=50), width=900, height=680,
        )
        try:
            fig.show(renderer="png", width=900, height=680, scale=2)
        except (ValueError, ImportError) as e:
            print(f"(Plotly PNG export unavailable — {e}; falling back to matplotlib)")
            self._show_s2_sphere_static_mpl()

        pt = pd["pos_thresholded"]
        print(
            f"Active PDs: {pt.shape[1]}  |  Total bins: {len(pd['occupancy_full'])}  |  "
            f"Total particles: {pd['pos_full'].shape[1]}  "
            "[static view — set INTERACTIVE=True for clickable diamonds]"
        )

    def _show_s2_sphere_static_mpl(self):
        """Matplotlib fallback for environments without kaleido."""
        pd = self.pd_data
        bc = pd["bin_centers"]
        occ = pd["occupancy_full"]
        pt = pd["pos_thresholded"]

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
        ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                        color="lightgrey", alpha=0.10, linewidth=0)
        sc = ax.scatter(bc[0], bc[1], bc[2], c=occ, cmap="viridis", s=6, alpha=0.85)
        ax.scatter(pt[0], pt[1], pt[2], c="red", s=55, marker="D",
                   edgecolors="darkred", linewidth=0.6,
                   label=f"Active PDs (n={pt.shape[1]})")
        cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.05)
        cb.set_label("Occupancy (particles/bin)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"S2 Orientation Distribution — {self.project_name}")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_box_aspect((1, 1, 1))
        plt.tight_layout()
        plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # Per-PrD eigenvector viewer
    # ─────────────────────────────────────────────────────────────────────
    def show_prd_viewer(self, default_prd: int = 1, interactive: bool = True):
        if interactive:
            self._show_prd_viewer_interactive(default_prd)
        else:
            self._render_prd_static(default_prd)

    def _gif_html(self, path: Path, width: int = 180) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return (f'<div style="display:inline-block;margin:4px;text-align:center">'
                f'<img src="data:image/gif;base64,{b64}" width="{width}"><br>'
                f'<small>{path.stem}</small></div>')

    def _render_prd_scree(self, prd_num: int):
        prd_dir = self.topos_dir / f"PrD_{prd_num}"
        eig_file = prd_dir / "eig_spec.txt"
        if not eig_file.exists():
            print(f"eig_spec.txt not found: {eig_file}")
            return
        data = np.loadtxt(eig_file)
        eig_idx = data[:, 0].astype(int)
        eig_val = data[:, 1]
        n_gifs = len(list(prd_dir.glob("psi_*.gif")))

        fig_s, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(eig_idx, eig_val, color="steelblue", edgecolor="none")
        ax.bar(eig_idx[:n_gifs], eig_val[:n_gifs], color="crimson",
               edgecolor="none", label=f"psi with movies (1–{n_gifs})")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Eigenvalue spectrum  —  PrD {prd_num}", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    def _render_prd_movies(self, prd_num: int):
        from IPython.display import HTML, display
        prd_dir = self.topos_dir / f"PrD_{prd_num}"
        gifs = sorted(prd_dir.glob("psi_*.gif"),
                      key=lambda p: int(p.stem.split("_")[1]))
        if not gifs:
            print(f"No .gif files found in {prd_dir}")
            return
        html_parts = [f'<h4 style="margin:6px 0">PrD {prd_num} — eigenvector movies</h4>']
        html_parts += [self._gif_html(g) for g in gifs]
        display(HTML("".join(html_parts)))

    def _render_prd_static(self, prd_num: int):
        """Render the scree plot + a static grid of the eigenvector movie thumbnails (first frame)."""
        from PIL import Image
        prd_dir = self.topos_dir / f"PrD_{prd_num}"
        self._render_prd_scree(prd_num)

        gifs = sorted(prd_dir.glob("psi_*.gif"),
                      key=lambda p: int(p.stem.split("_")[1]))
        if not gifs:
            print(f"No .gif files found in {prd_dir}")
            return
        n = len(gifs)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_2d(axes).reshape(rows, cols)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, gif in zip(axes.ravel(), gifs):
            with Image.open(gif) as im:
                ax.imshow(np.asarray(im.convert("L")), cmap="gray")
            ax.set_title(gif.stem, fontsize=9)
        plt.suptitle(f"PrD {prd_num} — eigenvector movies (first frame)\n"
                     "[static view — set INTERACTIVE=True for animated GIFs]",
                     fontsize=11)
        plt.tight_layout()
        plt.show()

    def _show_prd_viewer_interactive(self, default_prd: int = 1):
        import ipywidgets as widgets
        from IPython.display import display

        pd = self.pd_data
        n_prds = pd["pos_thresholded"].shape[1]
        phi_thres = pd["phi_thresholded"]
        theta_thres = pd["theta_thresholded"]
        occupancy = pd["occupancy_full"]
        thres_ids = list(pd["thres_ids"])

        prd_slider = widgets.IntSlider(
            value=default_prd, min=1, max=n_prds, step=1, description="PrD:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="420px"),
        )
        prd_info = widgets.Label(value="", layout=widgets.Layout(width="400px"))
        self._prd_slider = prd_slider

        scree_out = widgets.Output()
        movies_out = widgets.Output()
        tabs = widgets.Tab(children=[scree_out, movies_out])
        tabs.set_title(0, "Scree plot")
        tabs.set_title(1, "Eigenvector movies")

        def render(change):
            prd_num = prd_slider.value
            pd_idx = prd_num - 1
            prd_info.value = (
                f"PrD {prd_num}  |  φ={phi_thres[pd_idx]:.1f}°  "
                f"θ={theta_thres[pd_idx]:.1f}°  |  "
                f"occ = {occupancy[thres_ids[pd_idx]]}"
            )
            with scree_out:
                scree_out.clear_output(wait=True)
                self._render_prd_scree(prd_num)
            with movies_out:
                movies_out.clear_output(wait=True)
                self._render_prd_movies(prd_num)

        prd_slider.observe(render, names="value")
        render(None)

        display(widgets.VBox([widgets.HBox([prd_slider, prd_info]), tabs]))

    # ─────────────────────────────────────────────────────────────────────
    # Diffusion map analysis
    # ─────────────────────────────────────────────────────────────────────
    def show_diffusion_map(self, default_prd: int = 1, interactive: bool = True):
        if interactive and self._prd_slider is not None:
            self._show_diffusion_map_interactive()
        else:
            self._render_diffusion_map(default_prd, static_note=not interactive)

    def _render_diffusion_map(self, prd_num: int, static_note: bool = False):
        pd_idx = prd_num - 1
        dist_file = self.output_dir / "distances" / f"IMGs_prD_{pd_idx}.h5"
        diffmap_file = self.output_dir / "diff_maps" / f"gC_trimmed_psi_prD_{pd_idx}.h5"

        missing = [p for p in (dist_file, diffmap_file) if not p.exists()]
        if missing:
            for p in missing:
                print(f"Not found: {p}")
            return

        with h5py.File(dist_file, "r") as f:
            D = f["D"][:]
            rotations = f["rotations"][:]
        with h5py.File(diffmap_file, "r") as f:
            psi = f["psi"][:]
            logEps = f["logEps"][:]
            logSumWij = f["logSumWij"][:]
            popt = f["popt"][:]
            sigma = float(f["sigma"][()])
            R_sq = float(f["R_squared"][()])
            mu = f["mu"][:]
            posPath = f["posPath"][:]

        N = D.shape[0]
        logEps_opt = -popt[1] / popt[0]
        dim_est = 2 * popt[0] * popt[2]
        tanh_fit = popt[3] + popt[2] * np.tanh(popt[0] * logEps + popt[1])
        D_ord = D[np.ix_(posPath, posPath)]

        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.8))
        im = axes1[0].imshow(D_ord, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=axes1[0], fraction=0.046, pad=0.04)
        axes1[0].set_title(f"Distance matrix (posPath order)  —  PrD {prd_num}  (N={N})", fontsize=11)
        axes1[0].set_xlabel("Particle index (posPath)")
        axes1[0].set_ylabel("Particle index (posPath)")

        axes1[1].plot(logEps, logSumWij, ".", color="steelblue", ms=3, alpha=0.6, label="data")
        axes1[1].plot(logEps, tanh_fit, "-", color="limegreen", lw=2, label="tanh fit")
        axes1[1].axvline(logEps_opt, color="orange", lw=1.5, ls="--", label=f"ln ε* = {logEps_opt:.3f}")
        slope = popt[0] * popt[2]
        fit_at_opt = popt[3] + popt[2] * np.tanh(0.0)
        x_tang = np.array([logEps_opt - 3, logEps_opt + 3])
        axes1[1].plot(x_tang, fit_at_opt + slope * (x_tang - logEps_opt),
                      "r--", lw=1, alpha=0.7, label=f"slope = {slope:.3f}")
        ymin, ymax = logSumWij.min(), logSumWij.max()
        yspan = ymax - ymin or 1.0
        axes1[1].text(
            logEps.min() + 0.03 * (logEps.max() - logEps.min()),
            ymax - 0.05 * yspan,
            f"σ = {sigma:.3e}\nDim ≈ {dim_est:.2f}\nR² = {R_sq:.4f}",
            fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85),
        )
        axes1[1].set_xlabel("ln ε", fontsize=12)
        axes1[1].set_ylabel("ln Σ Wᵢⱼ", fontsize=12)
        axes1[1].set_title(f"Ferguson plot  —  PrD {prd_num}", fontsize=11)
        axes1[1].legend(fontsize=9)
        plt.tight_layout()
        plt.show()

        psi1 = psi[:, 0]
        psi2 = psi[:, 1]
        colorings = [
            (posPath.astype(float), "posPath (conformational order)", "viridis"),
            (mu, "μ (Riemannian density)", "plasma"),
            (rotations, "In-plane rotation (rad)", "RdBu"),
        ]
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, (c, clabel, cmap) in zip(axes2, colorings):
            sc = ax.scatter(psi1, psi2, c=c, cmap=cmap, s=14, alpha=0.75, linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=clabel)
            ax.set_xlabel("ψ₁", fontsize=11)
            ax.set_ylabel("ψ₂", fontsize=11)
            suffix = f"  —  {clabel}\nPrD {prd_num}"
            ax.set_title(f"ψ₁ vs ψ₂{suffix}", fontsize=10)
        plt.tight_layout()
        plt.show()

        if static_note:
            print(f"[static view — showing PrD {prd_num}. "
                  "Set INTERACTIVE=True to drive this with the PrD slider.]")

    def _show_diffusion_map_interactive(self):
        import ipywidgets as widgets
        from IPython.display import display

        dm_out = widgets.Output()

        def render(change):
            prd_num = self._prd_slider.value
            with dm_out:
                dm_out.clear_output(wait=True)
                self._render_diffusion_map(prd_num)

        self._prd_slider.observe(render, names="value")
        render(None)
        display(dm_out)

    # ─────────────────────────────────────────────────────────────────────
    # CC voting
    # ─────────────────────────────────────────────────────────────────────
    def show_cc_voting(self, interactive: bool = True):
        cc = self.cc_data
        psinums_all = cc["psinums"][0]
        senses_all = cc["senses"][0]
        n_prd_cc = len(psinums_all)
        prd_nums = np.arange(1, n_prd_cc + 1)
        colors_sense = ["tomato" if s > 0 else "steelblue" for s in senses_all]

        fig_cc, axes_cc = plt.subplots(1, 2, figsize=(14, 4.5))
        axes_cc[0].bar(prd_nums, psinums_all + 1, color=colors_sense, edgecolor="none", width=0.85)
        axes_cc[0].set_xlabel("PrD number", fontsize=12)
        axes_cc[0].set_ylabel("Chosen ψ index (1-indexed)", fontsize=12)
        axes_cc[0].set_title("CC voting — chosen ψ per PrD", fontsize=12)
        axes_cc[0].set_yticks(range(1, 9))
        axes_cc[0].set_yticklabels([f"ψ{k}" for k in range(1, 9)])
        axes_cc[0].legend(
            handles=[Patch(fc="tomato", label="FWD (+1)"),
                     Patch(fc="steelblue", label="REV (−1)")],
            fontsize=10, loc="upper right",
        )

        axes_cc[1].bar(prd_nums, senses_all, color=colors_sense, edgecolor="none", width=0.85)
        axes_cc[1].axhline(0, color="k", lw=0.8, ls="--")
        axes_cc[1].set_xlabel("PrD number", fontsize=12)
        axes_cc[1].set_ylabel("Sense (+1 FWD / −1 REV)", fontsize=12)
        axes_cc[1].set_title("CC voting — FWD / REV sense per PrD", fontsize=12)
        axes_cc[1].set_yticks([-1, 0, 1])
        axes_cc[1].set_yticklabels(["−1 REV", "0", "+1 FWD"])

        plt.suptitle(f"SENSE quality  —  {self.project_name}", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.show()

        if interactive and self._sphere_fig is not None:
            with self._sphere_fig.batch_update():
                self._sphere_fig.data[2].marker.color = psinums_all + 1
                self._sphere_fig.data[2].marker.colorscale = "Turbo"
                self._sphere_fig.data[2].marker.colorbar = dict(
                    title="Chosen ψ", thickness=14, x=1.20,
                    tickvals=list(range(1, 9)),
                    ticktext=[f"ψ{k}" for k in range(1, 9)],
                )

        unique_psis = dict(zip(*np.unique(psinums_all + 1, return_counts=True)))
        print(f"CC data loaded: {n_prd_cc} PrDs  |  ψ choices: {unique_psis}")

    # ─────────────────────────────────────────────────────────────────────
    # Optical flow
    # ─────────────────────────────────────────────────────────────────────
    def show_optical_flow(self, default_prd: int = 1, default_psi: int = 1,
                          interactive: bool = True):
        if interactive:
            self._show_optical_flow_interactive(default_prd, default_psi)
        else:
            self._render_optical_flow(default_prd, default_psi, static_note=True)

    def _render_optical_flow(self, prd_num: int, psi_num: int, static_note: bool = False):
        pd_idx = prd_num - 1
        psi_idx = psi_num - 1
        psinums_all = self.cc_data["psinums"][0]
        chosen_psi = int(psinums_all[pd_idx])
        star = " ★" if psi_idx == chosen_psi else ""

        of_file = self.output_dir / "CC" / "CC_OF" / f"OF_prD_{pd_idx}.pkl"
        img_path = self.topos_dir / f"PrD_{prd_num}" / "class_avg.png"

        if not of_file.exists():
            print(f"OF file not found: {of_file}")
            return
        with open(of_file, "rb") as f:
            of_data = pickle.load(f)
        fv = of_data["FlowVecPrD"]
        if psi_idx >= len(fv) or fv[psi_idx] is None:
            print(f"No OF data for ψ{psi_num} at PrD {prd_num}")
            return

        entry = fv[psi_idx]
        step = 16
        H, W = 336, 336
        ys = np.arange(step // 2, H, step)
        xs = np.arange(step // 2, W, step)
        Xg, Yg = np.meshgrid(xs, ys)

        fig_of, axes_of = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, direction, label in zip(axes_of, ["FWD", "REV"],
                                        ["FWD (τ: 0→1)", "REV (τ: 1→0)"]):
            if direction not in entry:
                ax.text(0.5, 0.5, f"{direction} data missing", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            vec = entry[direction]
            vx = np.asarray(vec["Vx"], dtype=float)
            vy = np.asarray(vec["Vy"], dtype=float)
            mag = np.asarray(vec["Mag"], dtype=float)

            if img_path.exists():
                bg = plt.imread(str(img_path))
                ax.imshow(bg, cmap="gray" if bg.ndim == 2 else None,
                          extent=[0, W, H, 0], origin="upper", alpha=0.6)
            else:
                ax.imshow(mag, cmap="hot", extent=[0, W, H, 0], origin="upper", alpha=0.4)

            u_sub = vx[Yg, Xg]
            v_sub = -vy[Yg, Xg]
            m_sub = np.sqrt(u_sub ** 2 + v_sub ** 2)
            ref_mag = float(np.percentile(m_sub[m_sub > 0], 95)) if (m_sub > 0).any() else 1e-9
            display_scale = (step * 0.70) / max(ref_mag, 1e-9)
            U = u_sub * display_scale
            V = v_sub * display_scale

            ax.quiver(Xg, Yg, U, V, mag[Yg, Xg],
                      cmap="plasma", scale=1.0, scale_units="xy", angles="xy",
                      width=0.004, headwidth=4, headlength=5, minlength=0.4)
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            ax.set_title(f"{label}   PrD {prd_num}  ψ{psi_num}{star}", fontsize=11)
            ax.axis("off")

        plt.suptitle(f"Optical flow quiver  —  PrD {prd_num}  ψ{psi_num}{star}", fontsize=12)
        plt.tight_layout()
        plt.show()

        if static_note:
            print(f"[static view — showing PrD {prd_num}, ψ{psi_num}. "
                  "Set INTERACTIVE=True for the PrD/ψ sliders.]")

    def _show_optical_flow_interactive(self, default_prd: int = 1, default_psi: int = 1):
        import ipywidgets as widgets
        from IPython.display import display

        prd_slider = self._prd_slider
        if prd_slider is None:
            n_prds = self.pd_data["pos_thresholded"].shape[1]
            prd_slider = widgets.IntSlider(
                value=default_prd, min=1, max=n_prds, step=1, description="PrD:",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="420px"),
            )
            self._prd_slider = prd_slider

        psi_slider = widgets.IntSlider(
            value=default_psi, min=1, max=8, step=1, description="ψ:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="320px"),
        )
        of_out = widgets.Output()

        def render(change):
            with of_out:
                of_out.clear_output(wait=True)
                self._render_optical_flow(prd_slider.value, psi_slider.value)

        prd_slider.observe(render, names="value")
        psi_slider.observe(render, names="value")
        render(None)

        display(widgets.VBox([widgets.HBox([prd_slider, psi_slider]), of_out]))

    # ─────────────────────────────────────────────────────────────────────
    # Trajectory free-energy (no widgets — always static)
    # ─────────────────────────────────────────────────────────────────────
    def show_trajectory(self, interactive: bool = True):  # interactive kept for API symmetry
        td = self.traj_data
        hUn = td["main"]["hUn"].astype(float)
        tauAvg = td["vars"]["tauAvg"]
        n_states = len(hUn)
        state_ax = np.arange(1, n_states + 1)

        prob = hUn / hUn.sum()
        prob_safe = np.where(prob > 0, prob, np.nan)
        dG = -np.log(prob_safe)
        dG -= np.nanmin(dG)

        tau_kde = stats.gaussian_kde(tauAvg, bw_method=0.05)
        tau_grid = np.linspace(0, 1, 200)
        tau_pdf = tau_kde(tau_grid)

        fig_prob, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        ax1.bar(state_ax, hUn, color="steelblue", edgecolor="none", width=0.85, alpha=0.85)
        ax1.set_xlabel("Conformational state (τ bin)", fontsize=12)
        ax1.set_ylabel("Particle count  hUn", fontsize=12)
        ax1.set_title(f"State histogram  —  {self.project_name}\n"
                      f"Total particles: {int(hUn.sum())}  |  50 states", fontsize=11)

        ax2.plot(state_ax, dG, "o-", color="crimson", lw=2, ms=5,
                 label=r"$-\ln P(\tau)$  [k$_B$T]")
        ax2.set_xlabel("Conformational state (τ bin)", fontsize=12)
        ax2.set_ylabel(r"$\Delta G(\tau)$  [k$_B$T]", fontsize=12, color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")

        ax2b = ax2.twinx()
        ax2b.plot(tau_grid * (n_states - 1) + 1, tau_pdf, "--", color="navy",
                  lw=1.5, alpha=0.75, label="tauAvg KDE")
        ax2b.set_ylabel("tauAvg KDE density", fontsize=11, color="navy")
        ax2b.tick_params(axis="y", labelcolor="navy")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper center")
        ax2.set_title("Free-energy landscape  —  −ln P(τ)", fontsize=11)

        plt.tight_layout()
        plt.show()
        print(f"hUn: min={hUn.min():.0f}  max={hUn.max():.0f}  total={hUn.sum():.0f}")
