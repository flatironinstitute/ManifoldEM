"""Plotting helpers for the ManifoldEM tutorial notebook.

`load_session(toml_path)` loads a project from its `params_<project>.toml` file and returns
a `VizSession`; project-specific values (such as `num_psi` and `states_per_coord`) come from
that file. Each `VizSession.show_*` method renders either a live ipywidgets/Plotly
viewer (`interactive=True`) or a static matplotlib snapshot that GitHub's
notebook viewer can display (`interactive=False`).

The `set_*`, `remove_anchor`, `trash_prd` and `untrash_prd` methods edit the
project between CLI steps: occupancy thresholds, anchors, trash PrDs and the
trajectory path width. They save through ManifoldEM's `params` and
`data_store`, so the CLI picks the changes up. PrD and ψ numbers are 1-based,
as in the `topos/PrD_N/psi_K.gif` movies.
"""
from __future__ import annotations

import base64
import pickle
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats


# The directory the kernel started in (the notebook's, under Jupyter). Relative project paths are
# resolved against it, because loading a project changes into the project's directory.
_START_DIR = Path.cwd()


def load_session(toml_path) -> "VizSession":
    """Load the project whose params file is `toml_path`, and return a VizSession.

    `toml_path` is the `params_<project>.toml` that `manifold-cli init` wrote. A relative path is
    relative to the directory the notebook was started in. Like the CLI, this changes into the
    project's directory.
    """
    from ManifoldEM.params import params

    toml_path = (_START_DIR / Path(toml_path).expanduser()).resolve()
    if not toml_path.is_file():
        raise FileNotFoundError(f"No project file at {toml_path}; set PROJECT_TOML to your params_<project>.toml.")
    print(f"Using TOML file: {toml_path}")

    params.load(str(toml_path))
    project_name = params.project_name
    print(f"project_name: {project_name}")

    output_dir = toml_path.parent / params.out_dir
    pkl_path = output_dir / "pd_data.pkl"
    if not pkl_path.exists():
        # Right after `manifold-cli init` the particles haven't been binned yet.
        print(f"{pkl_path} not found; binning the particles into projection directions…")
        _load_project(toml_path)
    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        pd_data = pickle.load(f)

    print(f"\nLoaded keys: {list(pd_data.keys())[:6]}…")
    print(f"  bin_centers       : {pd_data['bin_centers'].shape}")
    print(f"  occupancy_full    : {pd_data['occupancy_full'].shape}")
    print(f"  pos_thresholded   : {pd_data['pos_thresholded'].shape}")
    print(f"  thres_low / high  : {pd_data['thres_low']} / {pd_data['thres_high']}")

    return VizSession(pd_data=pd_data, output_dir=output_dir, project_name=project_name,
                      toml_path=toml_path, num_psi=params.num_psi,
                      states_per_coord=params.states_per_coord, traj_name=params.traj_name)


def _load_project(toml_path: Path):
    """Load the project into ManifoldEM's `params` and `data_store` singletons.

    Builds `pd_data.pkl` if it is missing or out of date with the thresholds, then re-reads it,
    so that edits saved elsewhere (the CLI, another kernel) are picked up. Like the CLI, this
    changes into the directory of the project file.
    """
    from ManifoldEM.data_store import data_store
    from ManifoldEM.params import params

    params.load(str(toml_path))
    prds = data_store.get_prds()
    prds.load()
    return params, prds


class VizSession:
    """Holds tutorial state shared across notebook cells.

    Cross-cell interactivity (sphere click → updates PrD slider, CC voting
    recolours sphere diamonds) requires sharing widget references; this
    object owns that state so the notebook cells stay one-liners.
    """

    def __init__(self, pd_data: dict, output_dir: Path, project_name: str, toml_path: Path,
                 num_psi: int, states_per_coord: int, traj_name: str):
        self.pd_data = pd_data
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.toml_path = Path(toml_path)
        self.num_psi = num_psi
        self.states_per_coord = states_per_coord
        self.traj_name = traj_name
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
            with open(traj_dir / f"traj_name{self.traj_name}.pkl", "rb") as f:
                main = pickle.load(f)
            with open(traj_dir / f"traj_name{self.traj_name}_vars.pkl", "rb") as f:
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

        # posPath lists the images kept after trimming; row i of psi and mu belongs to image posPath[i]
        N = D.shape[0]
        logEps_opt = -popt[1] / popt[0]
        dim_est = 2 * popt[0] * popt[2]
        tanh_fit = popt[3] + popt[2] * np.tanh(popt[0] * logEps + popt[1])
        by_psi1 = posPath[np.argsort(psi[:, 0])]
        D_ord = D[np.ix_(by_psi1, by_psi1)]

        fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.8))
        im = axes1[0].imshow(D_ord, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=axes1[0], fraction=0.046, pad=0.04)
        axes1[0].set_title(f"Distance matrix, sorted by ψ₁  —  PrD {prd_num}\n"
                           f"({len(posPath)} of {N} images kept after trimming)", fontsize=11)
        axes1[0].set_xlabel("Image (sorted by ψ₁)")
        axes1[0].set_ylabel("Image (sorted by ψ₁)")

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
        rot_kept = np.mod(rotations[posPath] + 180, 360) - 180  # degrees, wrapped to [-180, 180)
        colorings = [
            (psi[:, 2], "ψ₃", "viridis", None, None),
            (mu, "μ (Riemannian density)", "plasma", None, None),
            (rot_kept, "In-plane rotation (°)", "twilight", -180, 180),
        ]
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, (c, clabel, cmap, vmin, vmax) in zip(axes2, colorings):
            sc = ax.scatter(psi1, psi2, c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=14, alpha=0.75, linewidths=0)
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
        # find-ccs writes ψ = -1, sense 0 for PrDs without a coordinate (trashed, isolated, or no usable ψ)
        unassigned = prd_nums[psinums_all < 0]

        fig_cc, axes_cc = plt.subplots(1, 2, figsize=(14, 4.5))
        axes_cc[0].bar(prd_nums, psinums_all + 1, color=colors_sense, edgecolor="none", width=0.85)
        axes_cc[0].set_xlabel("PrD number", fontsize=12)
        axes_cc[0].set_ylabel("Chosen ψ index (1-indexed)", fontsize=12)
        axes_cc[0].set_title("CC voting — chosen ψ per PrD", fontsize=12)
        axes_cc[0].set_yticks(range(1, self.num_psi + 1))
        axes_cc[0].set_yticklabels([f"ψ{k}" for k in range(1, self.num_psi + 1)])
        handles = [Patch(fc="tomato", label="FWD (+1)"), Patch(fc="steelblue", label="REV (−1)")]
        if len(unassigned):
            for ax in axes_cc:
                ax.plot(unassigned, np.zeros(len(unassigned)), "x", color="grey", ms=8, mew=2, clip_on=False)
            handles.append(Line2D([], [], ls="", marker="x", color="grey", ms=8, mew=2, label="no coordinate"))
        axes_cc[0].legend(handles=handles, fontsize=10, loc="upper right")

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
                    tickvals=list(range(1, self.num_psi + 1)),
                    ticktext=[f"ψ{k}" for k in range(1, self.num_psi + 1)],
                )

        psis, counts = np.unique(psinums_all[psinums_all >= 0] + 1, return_counts=True)
        unique_psis = {int(k): int(n) for k, n in zip(psis, counts)}
        print(f"CC data loaded: {n_prd_cc} PrDs  |  ψ choices: {unique_psis}")
        if len(unassigned):
            print(f"No coordinate: PrD {', '.join(map(str, unassigned))}")

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
        H, W = np.shape(next(iter(entry.values()))["Vx"])  # the size of the movie frames
        step = max(1, W // 21)  # 16 px for a 336-px box
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
            value=default_psi, min=1, max=self.num_psi, step=1, description="ψ:",
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
    # Trajectory probability landscape (no widgets — always static)
    # ─────────────────────────────────────────────────────────────────────
    def show_trajectory(self, interactive: bool = True):  # interactive kept for API symmetry
        td = self.traj_data
        hUn = td["main"]["hUn"].astype(float)
        tauAvg = td["vars"]["tauAvg"]
        n_states = len(hUn)
        state_ax = np.arange(1, n_states + 1)

        prob = hUn / hUn.sum()
        prob_safe = np.where(prob > 0, prob, np.nan)
        neg_log_p = -np.log(prob_safe)
        neg_log_p -= np.nanmin(neg_log_p)

        tau_kde = stats.gaussian_kde(tauAvg, bw_method=0.05)
        tau_grid = np.linspace(0, 1, 200)
        tau_pdf = tau_kde(tau_grid)

        fig_prob, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        ax1.bar(state_ax, prob, color="steelblue", edgecolor="none", width=0.85, alpha=0.85)
        ax1.set_xlabel("Conformational state (τ bin)", fontsize=12)
        ax1.set_ylabel(r"Probability  $P(\tau)$", fontsize=12)
        ax1.set_title(f"Probability per state  —  {self.project_name}\n"
                      f"Total particles: {int(hUn.sum())}  |  {n_states} states", fontsize=11)

        ax2.plot(state_ax, neg_log_p, "o-", color="crimson", lw=2, ms=5,
                 label=r"$-\ln P(\tau)$")
        ax2.set_xlabel("Conformational state (τ bin)", fontsize=12)
        ax2.set_ylabel(r"$-\ln P(\tau)$  (relative to its minimum)", fontsize=12, color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")

        ax2b = ax2.twinx()
        ax2b.plot(tau_grid * (n_states - 1) + 1, tau_pdf, "--", color="navy",
                  lw=1.5, alpha=0.75, label="tauAvg KDE")
        ax2b.set_ylabel("tauAvg KDE density", fontsize=11, color="navy")
        ax2b.tick_params(axis="y", labelcolor="navy")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper center")
        ax2.set_title("Probability landscape  —  −ln P(τ)", fontsize=11)

        plt.tight_layout()
        plt.show()
        print(f"hUn: min={hUn.min():.0f}  max={hUn.max():.0f}  total={hUn.sum():.0f}")

    # ─────────────────────────────────────────────────────────────────────
    # Project edits between CLI steps (formerly made in the GUI):
    # occupancy thresholds, anchors, trash PrDs and trajectory path width
    # ─────────────────────────────────────────────────────────────────────
    def _project(self):
        return _load_project(self.toml_path)

    def _refresh(self):
        with open(self.output_dir / "pd_data.pkl", "rb") as f:
            self.pd_data = pickle.load(f)

    @staticmethod
    def _prd_index(prds, prd: int) -> int:
        if not 1 <= int(prd) <= prds.n_thresholded:
            raise ValueError(f"There is no PrD {prd}; PrD numbers run from 1 to {prds.n_thresholded}.")
        return int(prd) - 1

    def show_thresholds(self, low: Optional[int] = None, high: Optional[int] = None,
                        interactive: bool = True):  # interactive kept for API symmetry
        """Plot every S2 bin's occupancy against the low/high thresholds (the current ones by default).

        Bins with at least `low` particles become PrDs, and each PrD keeps at most `high` of them.
        Passing `low`/`high` only previews them; `set_thresholds` applies them.
        """
        pd = self.pd_data
        low = int(pd["thres_low"] if low is None else low)
        high = int(pd["thres_high"] if high is None else high)
        occ = np.asarray(pd["occupancy_full"])
        keep = occ >= low
        used = np.minimum(occ[keep], high)

        bc = pd["bin_centers"]
        phi = np.arctan2(bc[1], bc[0])
        theta = np.degrees(np.arccos(np.clip(bc[2], -1.0, 1.0)))

        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        occ_sorted = np.sort(occ)[::-1]
        rank = np.arange(1, len(occ) + 1)
        n_keep = int(keep.sum())
        ax1.plot(rank[:n_keep], occ_sorted[:n_keep], ".", ms=3, color="steelblue", label="become PrDs")
        ax1.plot(rank[n_keep:], occ_sorted[n_keep:], ".", ms=3, color="lightgrey", label="below low")
        ax1.axhline(low, color="red", lw=1, ls="--", label=f"low = {low}")
        ax1.axhline(high, color="green", lw=1, ls="--", label=f"high = {high}")
        ax1.set_xlabel("S2 bin (sorted by occupancy)")
        ax1.set_ylabel("Occupancy (particles/bin)")
        ax1.set_title(f"{n_keep} of {len(occ)} bins become PrDs", fontsize=11)
        ax1.legend(fontsize=9)

        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        ax2.scatter(phi[~keep], theta[~keep], s=4, color="lightgrey", linewidths=0)
        sc = ax2.scatter(phi[keep], theta[keep], s=14, c=occ[keep], cmap="viridis",
                         edgecolors="k", linewidths=0.3)
        plt.colorbar(sc, ax=ax2, pad=0.1, label="Occupancy (particles/bin)")
        ax2.set_ylim(0, 180)
        ax2.set_title("PrDs on S2  (angle φ, radius θ in degrees)", fontsize=11)
        plt.suptitle(f"Occupancy thresholds  —  {self.project_name}", fontsize=12)
        plt.tight_layout()
        plt.show()

        print(f"low={low}, high={high}: {n_keep} PrDs use {used.sum()} of {occ.sum()} particles "
              f"({occ[~keep].sum()} are in bins below low, {(occ[keep] - used).sum()} are over high)")
        if (low, high) != (pd["thres_low"], pd["thres_high"]):
            print(f"[preview only: the project uses low={pd['thres_low']}, high={pd['thres_high']}. "
                  "Apply with viz.set_thresholds(low, high).]")

    def set_thresholds(self, low: int, high: int):
        """Apply new occupancy thresholds, like `manifold-cli threshold --prd_thres_low … --prd_thres_high …`.

        The particles are re-binned right away, which renumbers the PrDs: all anchors and trash
        PrDs are cleared, and every step from calc-distance on has to be run again.
        """
        from ManifoldEM.params import ProjectLevel

        low, high = int(low), int(high)
        params, prds = self._project()
        if (low, high) == (prds.thres_low, prds.thres_high):
            print(f"The thresholds are already low={low}, high={high}; nothing changed.")
            return
        n_prds = int(np.sum(prds.occupancy_full >= low))
        if n_prds < 3:
            raise ValueError(f"Only {n_prds} bins have at least {low} particles; at least 3 PrDs are needed.")
        n_anchors, n_trash = len(prds.anchors), len(prds.trash_ids)

        params.prd_thres_low, params.prd_thres_high = low, high
        params.project_level = ProjectLevel.BINNING
        params.save(str(self.toml_path))
        _load_project(self.toml_path)  # re-bins because the thresholds changed
        self._refresh()

        print(f"Thresholds set to low={low}, high={high}: {n_prds} PrDs. "
              "Run calc-distance and the later steps (again).")
        if n_anchors or n_trash:
            print(f"Cleared the anchors ({n_anchors}) and trash PrDs ({n_trash}), since the PrD numbers changed.")

    def set_anchor(self, prd: int, psi: int, sense: str = "FWD"):
        """Anchor PrD `prd` for find-ccs: its motion is the ψ`psi` movie (psi_`psi`.gif) played `sense`.

        `sense` is "FWD" or "REV". Choose it so that every anchor shows the same physical motion
        (e.g. open → closed). Replaces any existing anchor on the PrD.
        """
        from ManifoldEM.data_store import Anchor, Sense

        params, prds = self._project()
        idx = self._prd_index(prds, prd)
        if not 1 <= int(psi) <= params.num_psi:
            raise ValueError(f"ψ must be between 1 and num_psi={params.num_psi}.")
        if isinstance(sense, str):
            if sense.upper() not in ("FWD", "REV"):
                raise ValueError('sense must be "FWD" or "REV".')
            sense = Sense[sense.upper()]
        else:
            sense = Sense(sense)
        if idx in prds.trash_ids:
            raise ValueError(f"PrD {prd} is in the trash; call untrash_prd({prd}) first.")

        prds.insert_anchor(idx, Anchor(int(psi), sense))
        prds.save()
        self._refresh()
        print(f"PrD {prd} anchored on ψ{psi} {sense.name}  ({len(prds.anchors)} anchors)")

    def remove_anchor(self, prd: int):
        """Remove the anchor from PrD `prd`, if it has one."""
        _, prds = self._project()
        idx = self._prd_index(prds, prd)
        if idx not in prds.anchors:
            print(f"PrD {prd} has no anchor.")
            return
        prds.remove_anchor(idx)
        prds.save()
        self._refresh()
        print(f"Removed the anchor from PrD {prd}  ({len(prds.anchors)} anchors)")

    def trash_prd(self, *prd_nums: int):
        """Leave PrDs out of find-ccs and everything after it (landscape and trajectories).

        Trashing a PrD also removes its anchor.
        """
        _, prds = self._project()
        for idx in [self._prd_index(prds, p) for p in prd_nums]:
            prds.trash_ids.add(idx)
            prds.remove_anchor(idx)
        prds.save()
        self._refresh()
        print(f"Trashed PrD {', '.join(map(str, prd_nums))}  ({len(prds.trash_ids)} in the trash)")

    def untrash_prd(self, *prd_nums: int):
        """Take PrDs out of the trash again."""
        _, prds = self._project()
        for idx in [self._prd_index(prds, p) for p in prd_nums]:
            prds.trash_ids.discard(idx)
        prds.save()
        self._refresh()
        print(f"Restored PrD {', '.join(map(str, prd_nums))}  ({len(prds.trash_ids)} in the trash)")

    def show_anchors(self, interactive: bool = True):  # interactive kept for API symmetry
        """Show the anchors and trash PrDs on S2, and check that every connected component has an anchor.

        find-ccs propagates each anchor's ψ and sense along the PrD neighbour graph, with the trash
        PrDs removed. The PrDs of a component without an anchor get no conformational coordinate
        and are left out of the landscape and trajectories; with no anchors, find-ccs stops early.
        """
        params, prds = self._project()
        components = _graph_components(params, prds)
        anchors, trash = prds.anchors, sorted(prds.trash_ids)
        phi = np.radians(prds.phi_thresholded)
        theta = np.asarray(prds.theta_thresholded)

        fig = plt.figure(figsize=(10, 6.5))
        ax = fig.add_subplot(111, projection="polar")
        cmap = plt.get_cmap("tab10")
        for k, comp in enumerate(components):
            label = f"component {k + 1} ({len(comp)} PrDs)"
            if not any(i in anchors for i in comp):
                label += " — no anchor"
            ax.scatter(phi[comp], theta[comp], s=30, color=cmap(k % 10), linewidths=0, label=label)
        if trash:
            ax.scatter(phi[trash], theta[trash], s=40, marker="x", color="grey", label=f"trash ({len(trash)})")
        if anchors:
            ids = sorted(anchors)
            ax.scatter(phi[ids], theta[ids], s=220, marker="*", color="gold", edgecolors="k",
                       linewidths=0.8, zorder=3, label=f"anchors ({len(ids)})")
            for i in ids:
                ax.annotate(str(i + 1), (phi[i], theta[i]), textcoords="offset points",
                            xytext=(7, 5), fontsize=8)
        ax.set_ylim(0, 180)
        ax.set_title(f"Anchors and trash PrDs  —  {self.project_name}\n"
                     "(angle φ, radius θ in degrees; labels are PrD numbers)", fontsize=11)
        ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), fontsize=8)
        plt.tight_layout()
        plt.show()

        occupancy = prds.occupancy
        for k, comp in enumerate(components):
            here = [i for i in comp if i in anchors]
            if here:
                listed = ", ".join(f"PrD {i + 1} ψ{anchors[i].CC} {anchors[i].sense.name}" for i in here)
                print(f"component {k + 1:>2} ({len(comp):>3} PrDs): {listed}")
            else:
                top = max(comp, key=lambda i: occupancy[i])
                print(f"component {k + 1:>2} ({len(comp):>3} PrDs): NO ANCHOR  "
                      f"(most populated: PrD {top + 1}, {occupancy[top]} particles)")
        if trash:
            print(f"trash: PrD {', '.join(str(i + 1) for i in trash)}")
        if not anchors:
            print("No anchors yet: find-ccs needs at least one. Use viz.set_anchor(prd, psi, sense).")

    def set_path_width(self, width: int):
        """Set the trajectory path width `width_1D` for the next `manifold-cli trajectory` run.

        Each of the `states_per_coord - width + 1` output volumes pools the particles of `width`
        adjacent states along the conformational coordinate.
        """
        params, _ = self._project()
        if not 1 <= int(width) <= params.states_per_coord:
            raise ValueError(f"The width must be between 1 and states_per_coord={params.states_per_coord}.")
        previous = params.width_1D
        params.width_1D = int(width)
        params.save(str(self.toml_path))
        print(f"Path width set to {width} (was {previous}); it takes effect the next time the trajectory step runs.")
        if int(width) > previous and any((self.output_dir / "bin").glob("*.star")):
            print("A wider path writes fewer volumes, and the trajectory step doesn't delete the old ones: "
                  "clear bin/ before rerunning it.")


def _graph_components(params, prds) -> list:
    """Connected components of the PrD neighbour graph without the trash PrDs, as find-ccs sees them."""
    import contextlib
    import io
    from copy import deepcopy

    from ManifoldEM.FindCCGraph import prune

    G = prds.neighbor_graph
    if prds.trash_ids:
        with contextlib.redirect_stdout(io.StringIO()):  # prune() reports the graph sizes
            G, _ = prune(deepcopy(G), prds.trash_ids, params.num_psi)
    components = [sorted(int(i) for i in np.atleast_1d(c)) for c in G["NodesConnComp"]]
    return [c for c in components if not set(c) <= prds.trash_ids]
