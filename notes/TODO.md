* TODO anchor node bug (delete prd?)
  * This just requires a much deeper understanding of their data structures, which will be
    easier as refactoring occurs. On hold sadly.
* TODO NLSA.py cli
  * Less necessary now that I've added --nograph, but CLI still very useful
* TODO 2d landscape
  * Already some beta code here (set Beta=False in ManifoldEM_GUI)
* TODO remove .op ridiculousness
* TODO optimize FindConformationalCoord
* TODO optimize "PD Thresholding"
* TODO stop using list expansion for arguments
  * This ain't matlab.
* TODO Compress IMGs_prD (distances)
  * Given that images are signed float64s, first step might be to make float32 rather than scaled int
* TODO compress pngs
  * Probably just hold them in a zip archive to reduce filecount
* TODO Switch out progress tracking technique
* TODO Low priority things
  * start adding types to all the things
  * remove duplicate code in ManifoldEM_GUI
