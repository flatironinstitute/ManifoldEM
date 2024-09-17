Changes made prior to the Flatiron fork not included.

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
    * Unify parameters implementation
    * Automatically generate help information for CLI
    * Add central store for project metadata (information on rotation, prd indices, image indices, etc)
    * Conjugate images are now handled via transformation flags and halving S2, rather than duplication. I.e. the number of images directly maps to the input stack, rather than relying on duplication and deduplication logic
