# manifold-cli init

* create `params_project.toml` -- [text] project parameters

# manifold-cli threshold

* modify `params_projects.toml`
  * `PDsizeThH` - high threshold (max images to analyze per bin)
  * `PDsizeThL` - low threshold (minimum number of images to consider valid bin)

# manifold-cli calc-distance

* create `pd_data` -- [pickle] aggregate/common data related to PDs. Recommend access through `ManifoldEM.data_store`
  * `thres_low` - [int] cached version of params PDsizeThL - if mismatch with params, rebuilds connectivity graph
  * `thres_high` - [int] cached version of params PDsizeThH - if mismatch with params, rebuilds connectivity graph
  * `bin_centers` - [3 x n_bins, float] cartesian location of bins for tesselated S2 surface
  * `defocus` - [~2 * n_images, float] duplicated defocus values from align data
  * `microscope_origin` - [2 x n_images, float] unduplicated (x,y) `sh` param of image origins
  * `pos_full` - [3 x ~2*n_images, float] duplicated cartesian positions of images in S2
  * `quats_full` - [4 x ~2*n_images, float] duplicated quatertions of images with surface rotations in S2
  * `image_indices_full` - [n\_bins x images\_in\_bin[i], int] jagged array of raw image duplicated image indices in bin `i`
  * `thres_ids` - [n_prds, int] bin id of each prd that met the `thres_low` qualification
  * `occupancy_full` - [n_bins, int] count of images in each bin
  * `anchors` - [dict[int, Anchor]] map prd index to anchor info
  * `trash_ids` - [set(int)] list of prd indices that we manually ignore
  * `pos_thresholded` - [3 x n_prds, float] cartesian S2 coordinates of each prd center
  * `phi_thresholded` - [n_prds, float] aximuthal angle of S2 coordinate for each prd center
  * `theta_thresholded` - [n_prds, float] polar angle of S2 coordinate for each prd center
  * `neighbor_graph` - [undocumented complicated dict]
  * `neighbor_subgrpah` - [undocumented complicated dict]
  * `cluster_ids` - [n_prds, int] cluster id (aka color) for each prd
* create `distances/IMGs_prD_{prd}` - [pickle] image distance data for each prd in bin
  * `D` - [images\_in\_bin[prd] x images\_in\_bin[prd], float] distance matrix for prd
  * `ind` - [images\_in\_bin[prd], int] relevant duplicated image indices to bin
  * `q` - [4 x images\_in\_bin[prd], float] quaternion 'position' on S2 with rotation for each image in bin
  * `CTF` - [images\_in\_bin[prd] x nPix**2, float] CTF for each image in bin
  * `imgAll` - [images\_in\_bin[prd] x nPix x nPix, float] processed image data for each image in bin
  * `msk2` - [Union[nPix x nPix, 1], int] image mask given avg orientation in bin. just `1` if no volumetric mask input
  * `PD` - [float] average orientation vector of images in bin
  * `imgAvg` - [nPix x nPix, float] wiener filtered average of `imgAll`
