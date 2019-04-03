# Invasion Counter

This GUI-based program allows a user to segment
a portion of the a spheroid as well as outerlying
cells in order to quantify cells migrating away from
a spheroid in 3D.


##  Dependencies

* [Python](www.python.org) written and tested in 2.7, untested in 3+
* [Numpy](www.numpy.org)
* [Scipy](www.scipy.org)
* [Matplotlib](www.matplotlib.org)
* [scikit-image](https://scikit-image.org/)
* [PyQt4](https://www.riverbankcomputing.com/software/pyqt/intro)
* A window system (e.g. XQuartz for OSX)

## Usage notes

Once the GUI is opened with `run_invasion_counter_gui.py`,
an image stack can be loaded (nuclei labeling for spheroid and invading cells).
The spheroid contour and invading nuclei centroids are segmented in turn,
and the results can be ouput to lists and plots. The positions of the invading
centroids can be updated/added/removed by hand and saved or re-loaded.

## Additional notes

Versions of this software have been used in the following publications:

- Attieh Y, **Clark, AG**, Grass C, Richon S, Pocard M, Mariani P, Elkhatib N Betz T, Gurchenkov B and Vignjevic DM (2017) Cancer-associated fibroblasts lead tumor invasion through integrin-β3-dependent fibronectin assembly. *J Cell Biol* 216(11):3509.
- Staneva R, Burla F, Koenderink GH, Descroix S, Vignjevic DM, Attieh Y, and Verhulsel, M ((2018). A new biomimetic assay reveals the temporal role of matrix stiffening in cancer cell invasion. *Mol Biol Cell* 29(25):2979.

This was developed in python 2.7.
I am not currently maintaining it,
and I know there will are a number minor bugs due to updated packages
(especially related to PyQt4, which is now PyQt5; also related to some
things updated in python 3.x).