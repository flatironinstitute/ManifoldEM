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
  * Added
    * Tutorial notebook helpers for what used to need the GUI: occupancy thresholds, anchors, trash
      PrDs and the trajectory path width (`viz.show_thresholds`, `viz.set_anchor`,
      `viz.trash_prd`, `viz.show_anchors`, `viz.set_path_width`, ...)
  * Changed
    * The tutorial notebook works for any project. It moved to `tutorial/`, its setup cell takes the
      path of the project's `.toml` file (`PROJECT_TOML`), and the number of psis and states, the
      trajectory name and the image size come from the project instead of the RyR1 tutorial
  * Fixed
    * **Changes results:** belief propagation used anchors off by one, so an anchor set on psi 1
      was propagated as psi 2 (and a REV anchor on the last psi crashed)
    * matplotlib is a core dependency now, so an install without extras can run the whole pipeline
    * Tutorial notebook plots:
      * The diffusion-map panel no longer fails on PrDs that lost images in trimming. It colours
        the images that were kept by their in-plane rotation, in degrees (not radians, as the label said).
      * The panel's distance matrix is sorted by psi 1. `posPath` is the list of images kept after
        trimming, not a conformational order.
      * The CC voting plot marks PrDs that have no coordinate. It used to leave them as gaps and
        count them as "psi 0".
      * The trajectory section shows the probability P and -ln P, not a free energy in k_BT
  * Outputs and project files
    * No longer written: `topos/PrD_*/psi_*.zip` frame archives (the GIFs remain), the empty
      `topos/Euler_PrD/` directory, and the unread keys of the trajectory `_vars.pkl` (only
      `tauAvg` remains)
    * Older project files still load: parameters that no longer exist (`num_part`, `width_2D`,
      `vis_s2_*`) are skipped silently

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
