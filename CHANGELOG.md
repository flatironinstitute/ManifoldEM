Changes made prior to the Flatiron fork not included.

* Unreleased
  * Removed
    * The GUI (`manifold-gui`) and the `[gui]` extra (PyQt5, mayavi, vtk, configobj, fabric), including
      running steps on a remote host over SSH, re-embedding a PrD, and the cluster average (which
      had been broken since the distance files stopped storing images). The pipeline runs from
      `manifold-cli`; the tutorial notebook inspects the results.
    * The average volume (`manifold-cli init -v`) is no longer used; only the GUI's S2 view read it
    * The `MANIFOLD_DISABLE_VIZ` environment variable, which only the GUI read
    * The unmaintained pre-2023 `data_viewers/` scripts, and the `opencv-contrib-python` dependency
      (`opencv-python-headless` has everything the pipeline uses)
    * The `numba` dependency, which only the replaced Euler-angle fit (`q2Spider`) used
    * The `calc-distance`, `manifold-analysis`, `psi-analysis` and `nlsa-movie` commands, which
      `decompose` replaces (see Changed)
    * The tau-based bad-psi detection, which had had no effect since 2023, and the parameters that
      only it or the replaced modules read: `find_bad_psi_tau`, `tau_occ_thresh`,
      `use_pruned_graph`, `calc_optical_flow`, `calc_all_edge_measures`, `opt_mask_type`,
      `opt_mask_param`, `n_reaction_coords`, `nlsa_fps` and `prd_n_active`
    * The `scikit-image` and `imageio` dependencies, which nothing uses any more
  * Added
    * Tutorial notebook helpers for what used to need the GUI: occupancy thresholds, anchors, trash
      PrDs and the trajectory path width (`viz.show_thresholds`, `viz.set_anchor`,
      `viz.trash_prd`, `viz.show_anchors`, `viz.set_path_width`, ...)
  * Changed
    * The pipeline is the one from Robert Blackwell's `in-memory` branch:
      * One `decompose` step replaces `calc-distance`, `manifold-analysis`, `psi-analysis` and
        `nlsa-movie`. For each projection direction it calculates the distances, embeds and trims
        the manifold, and makes the NLSA movie of each psi.
      * The results are in one file, `prd_analysis.h5`, with a group per projection direction:
        the distances, embeddings and movies, and after `find-ccs` the chosen psi and sense and the
        optical flow. It replaces `distances/`, `diff_maps/`, `psi_analysis/`, `topos/`, `CC/`,
        `ELConc*/` and `traj/`.
      * `probability-landscape` writes `probability_landscape.npz` (the occupancy of the states,
        and the taus it is the histogram of). `trajectory` writes `bin/` as before, under the same
        names.
      * `decompose` starts you off with anchors: for each connected component, psi 1 played forward
        on its most populated projection direction, unless that one already has an anchor. Check
        them before `find-ccs`.
      * Projects made by earlier versions have to be re-run from `decompose`. Their project files
        still load, and the steps that need `prd_analysis.h5` say so.
      * The steps are numbered 0-5 in `manifold-cli --help`.
    * Parameter flags on the steps after `init` (e.g. `manifold-cli decompose --num_psi 5`) now
      take effect and are saved to the project. Flags that aren't given leave the project alone:
      a bare `manifold-cli threshold` used to reset the thresholds and the hemisphere vector to
      their defaults.
    * `data_store.get_prd_data` reads `prd_analysis.h5`. `EL_data` is the NLSA data of the psi that
      `find-ccs` chose, and its movie `IMG1` is now n_frames x n_pix x n_pix.
    * The tutorial notebook reads `prd_analysis.h5`, and plays the psi movies from it. The
      `topos/` GIFs, the class averages and `eig_spec.txt` aren't written any more.
    * The tutorial notebook's plotting code is part of the package now, as
      `ManifoldEM.visualization` (`from ManifoldEM.visualization import load_session`). Its
      packages come with the new `notebook` extra (plotly, kaleido, ipywidgets, anywidget), which
      replaces the notebook's `%pip` cell. Without the extra, the static plots still work.
    * The tutorial notebook works for any project. It moved to `tutorial/`, its setup cell takes the
      path of the project's `.toml` file (`PROJECT_TOML`), and the number of psis and states, the
      trajectory name and the image size come from the project instead of the RyR1 tutorial
  * Fixed
    * **Changes results:** runs weren't reproducible. Each eigenvector of a diffusion map (and each
      singular vector of NLSA's decomposition) has an arbitrary sign, and which sign the solver
      returned could depend on rounding. The sign of a psi decides the direction of its movie and
      tau, so two runs on the same data could choose different coordinates and write different
      trajectories. Every eigenvector now has its largest entry positive. About half of the psis
      change sign against earlier runs. On the RyR1 tutorial subset, the choices change only in
      sense, where a psi's movie now plays the other way; on data with less structure, other psis
      can be chosen.
    * **Changes results:** belief propagation used anchors off by one, so an anchor set on psi 1
      was propagated as psi 2 (and a REV anchor on the last psi crashed)
    * **Changes results:** the Euler angles of a projection direction came from a least-squares fit
      (`q2Spider`) that started at gimbal lock. Depending on the state of the process, it sometimes
      stopped at its start point, and it always did within about 1e-5 of the hemisphere edge
      (x = 0). Particles it failed on were written to the trajectory `.star` files with rot = tilt = 0
      (a quarter of the particles in one test run). A PrD whose reference direction it failed on had
      all its images rotated by a wrong in-plane angle in the distance calculation. The angles are now
      computed in closed form. Where the fit worked, the results are the same to 1e-13 degrees. With
      a non-default `tess_hemisphere_vec`, PrDs at x < 0 now use the equivalent frame rotated by 180
      degrees (images and angles together), which also fixes PrDs whose particles straddle x = 0.
    * matplotlib is a core dependency now, so an install without extras can run the whole pipeline
    * Tutorial notebook helpers: with two projects loaded in one kernel, an edit could re-bin a
      project and so wipe its anchors and trash PrDs. `data_store.get_prds(reload=True)` re-reads
      `pd_data.pkl` of the project `params` points to.
    * Tutorial notebook: the CC voting, optical-flow and trajectory views kept showing the files as
      they were when first read, so re-running `find-ccs` or `trajectory` didn't show up until the
      setup cell ran again. Every view now reads the project again.
    * Tutorial notebook plots:
      * The diffusion-map panel no longer fails on PrDs that lost images in trimming. It colours
        the images that were kept by their in-plane rotation, in degrees (not radians, as the label said).
      * The panel's distance matrix is sorted by psi 1. `posPath` is the list of images kept after
        trimming, not a conformational order.
      * The CC voting plot marks PrDs that have no coordinate. It used to leave them as gaps and
        count them as "psi 0".
      * The trajectory section shows the probability P and -ln P, not a free energy in k_BT
      * On the interactive S2 sphere, CC voting coloured PrDs without a coordinate as the lowest
        psi. They're grey now, and each psi keeps the same colour whichever psis were chosen.
  * Outputs and project files
    * See Changed for the new layout (`prd_analysis.h5`, `probability_landscape.npz`)
    * Older project files still load: parameters that no longer exist (`num_part`, `width_2D`,
      `vis_s2_*` and those listed under Removed) are skipped silently

* main branch (since original 0.2.0-beta)
  * Usability
    * Make utility pip installable
    * Add command line utility (manifold-cli)
    * Make parameters human readable/editable (.toml)
    * Fix crashes when trashing entire PrD clusters
    * Remove MPI, only using python multithreading
    * Collect outputs more systematically, multiple projects in a single directory more feasable
    * Can disable 3D visualizations for running remotely via X forwarding (manifold-gui -V)
    * Optimize various plotting routines
    * User selectable division planes to exploit symmetry in image collection
    * Various bugfixes
  * Optimizations (Net ~10x or greater speedup across pipeline)
    * Considerable optimizations not listed below by using numba when helpful, more efficient
      math calculations with numpy, and changing out library calls for equivalent faster
      versions
    * Optimize rotate_fill routines for image rotation (~10x)
    * Replace scipy HOG (histogram of oriented gradients) routine with custom built fasthog library (~100x)
    * Parallelize every major step in pipeline
    * Replace Hornschunck convolution implementation (~5x)
    * Reduce storage requirements, both size and filecount
    * Considerably reduce file open/close operations
    * Use imageio rather than matplotlib for image output when applicable
  * Developer improvements
    * Delete thousands of lines of dead/redundant code
    * Improve tidiness of nearly every routine, with more consistent naming
    * Re-write GUI nearly from scratch (manifold-gui) to be more easily modifiable
    * Unify parameters implementation (remove code duplication in filenames -- easier to change where to store data)
    * Automatically generate help information for CLI
    * Add central store for project metadata (information on rotation, prd indices, image indices, etc)
    * Conjugate images are now handled via transformation flags and halving S2, rather than duplication. I.e. the number of images directly maps to the input stack, rather than relying on duplication and deduplication logic
    * Add thousands of lines of function documentation
    * Add type hints to many routines
