#!/opt/local/bin/python

__author__ = "Andrew G. Clark"
__date__ = "2015"
__copyright__ = "Copyright 2015, Andrew Clark"
__maintainer__ = "Andrew G. Clark"
__email__ = "andrew.clark@curie.fr"
__status__ = "Production"

""" A Qt GUI for the analysis of invasion from cellular spheroids.

"""

import sys

import numpy as np
from PyQt4 import QtGui, QtCore
from skimage.io._plugins import tifffile_plugin as tifffile

from image_io import Im, save_data_array, read_file, load_spheroid_image, load_nuclei_image, load_nuclei_list, save_nuclei_centroids
import find_spheroid_and_nuclei_v1 as seg
from invasion_counter_gui_v1_2 import Ui_TestMainWindow

class Simple(QtGui.QMainWindow):

    def __init__(self,parent=None):

        #super(Simple, self).__init__(parent)
        QtGui.QMainWindow.__init__(self)

        self.ui = Ui_TestMainWindow()
        self.ui.setupUi(self)
        self.initUI()

        self.ui.button_load_stk.clicked.connect(self.load_stk)
        self.ui.button_clear_data.clicked.connect(self.clear_data)
        self.ui.button_next.clicked.connect(self.next_slice)
        self.ui.button_previous.clicked.connect(self.prev_slice)
        self.ui.button_go.clicked.connect(self.go_to_slice)

        self.ui.button_find_nuclei.clicked.connect(self.find_nuclei)
        self.ui.button_load_nuclei.clicked.connect(self.load_nuclei)
        self.ui.button_save_nuclei.clicked.connect(self.save_nuclei)
        self.ui.button_find_spheroid.clicked.connect(self.find_spheroid)
        self.ui.button_load_spheroid.clicked.connect(self.load_spheroid)
        self.ui.button_save_spheroid.clicked.connect(self.save_spheroid)
        self.ui.button_plot_get_stats.clicked.connect(self.plot_get_stats)
        self.ui.button_save_settings.clicked.connect(self.save_settings)
        self.ui.button_load_settings.clicked.connect(self.load_settings)
        self.ui.button_default_settings.clicked.connect(self.restore_default_settings)
        self.ui.button_save_everything.clicked.connect(self.save_everything)
        self.ui.button_load_everything.clicked.connect(self.load_everything)
        self.ui.button_redraw.clicked.connect(self.draw_img)

        #declare instance variables
        self.im_path = None
        self.im = None
        self.scaling = None
        self.current_slice = None
        self.dot_pos_list_stk = None
        self.output_toggle = None
        self.stk_hist_sph = None
        self.sigma_sph = None
        self.thresh_type_sph = None
        self.thresh_factor_sph = None
        self.erosion_factor_sph = None
        self.smoothing_factor_sph = None
        self.stk_hist_nuclei = None
        self.sigma_nuclei = None
        self.thresh_type_nuclei = None
        self.thresh_factor_nuclei = None
        self.min_size_nuclei = None
        self.contour = None
        self.mask = None
        self.nuclei = None
        self.centroids = None
        self.closest_contour_pixels = None
        self.px_size_xy = 1.0
        self.px_size_z = 1.0

    def initUI(self):
        pass
        #self.ui.widget_img.fitInView(QtCore.QRectF(0,0,300,300),mode=1)

    def load_stk(self,open_path=False):

        # open_path = "./ctrl_1_small/ctrl_1_small.tif"

        if not open_path:
            #selects image path
            if not self.im_path is None:
                path_hint = self.im_path
            else:
                path_hint = './.tif'
            self.im_path = str(QtGui.QFileDialog.getOpenFileName(self, 'Select image stack', path_hint))
        else:
            self.im_path = open_path

        if len(self.im_path)>0:

            self.get_current_settings()

            #opens the image
            self.update_progress("Loading: %s"%self.im_path.split('/')[-1])
            self.im = Im(self.im_path,scaling=self.scaling)

            # #raise warning if the image is too large
            # if self.im.im_size[1] > self.ui.widget_img.height() or self.im.im_size[2] > self.ui.widget_img.width():
            #     QtGui.QMessageBox.about(self,"Title","Warning: your image is too large! " +
            #                             "Your image and centroid positions may not be displayed correctly! " +
            #                             "Please crop your image to a maximum of h = %i, w = %i pixels before proceeding!"
            #                             %(self.ui.widget_img.height(),self.ui.widget_img.width()))

            #prepares to display the image
            self.set_disp_params(self.im.im_size[1],self.im.im_size[2])
            self.current_slice = 1
            self.dot_pos_list_stk = [[] for _ in range(self.im.no_slices)]
            self.contour = np.zeros((self.im.no_slices,self.im.im_size[1],self.im.im_size[2]),dtype=np.uint8)

            #sends the first slice image to the display widget
            # print self.stk_array
            self.ui.widget_img.delete_items()
            self.draw_img()
            self.reset_progress()

            #reset some instance variables (to prevent using variables from previous mages)
            self.mask = None
            self.nuclei = None
            self.centroids = None
            self.closest_contour_pixels = None

    def load_spheroid(self,open_path=False):

        if not self.im is None:

            if not open_path:

                #gets the spheroid contour path
                if not self.im_path is None:
                    path_hint = self.im_path.replace(".tif", "_sph_contour.tif")
                else:
                    path_hint = './sph_contour.tif'
                open_path = str(QtGui.QFileDialog.getOpenFileName(self, 'Select spheroid contour image', path_hint))
                # contour_path = "./ctrl_1_small/ctrl_1_small_sph_contour.tif"

            if open_path:
                #opens and displays the spheroid contour image
                self.update_progress("Loading: %s"%open_path.split('/')[-1])
                self.contour, self.mask = load_spheroid_image(open_path)
                self.draw_img()
                self.update_progress("Spheroid Contour Loaded!")

        else:
            QtGui.QMessageBox.about(self,"Title","Please load an image file first!")


    def save_spheroid(self, save_path=False):

        if not self.contour is None:

            if not save_path:
                #gets the spheroid contour save path
                if not self.im_path is None:
                    path_hint = self.im_path.replace(".tif", "_sph_contour.tif")
                else:
                    path_hint = "./sph_contour.tif"
                save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Select save path for spheroid contour image',
                                                                  path_hint))

            if save_path:
                tifffile.imsave(save_path, 255*np.array(self.contour,dtype="uint8"))

        else:
            QtGui.QMessageBox.about(self,"Title","No spheroid contour to save!")

    def load_nuclei(self,open_path=False,ask=True):

        if self.centroids is None:
            ask=False

        answer = True
        if ask:
            answer = self.ask_permission("Do you really want to clear the current centroid data?")

        if answer==True:

            if not self.contour is None:

                if not open_path:

                    #gets the nuclei centroids save path
                    if not self.im_path is None:
                        path_hint = self.im_path.replace(".tif", "_nuclei_centroids.tif")
                    else:
                        path_hint = "./nuclei_centroids.tif"
                    open_path = str(QtGui.QFileDialog.getOpenFileName(self, 'Select nuclei centroid image or coordinate file',
                                                                          path_hint))

                if open_path:

                    if ".tif" in open_path:
                        self.update_progress("Loading nuclei centroids from image file...")
                        self.remove_centroids()
                        self.centroids, self.dot_pos_list_stk = load_nuclei_image(open_path)
                        self.update_progress("Determining Closest Contour Positions...")
                        # self.closest_contour_pixels = seg.get_closest_contour_px(self.contour,self.centroids)
                        self.draw_img()
                        self.update_progress("Nuclei centroids loaded!")

                    elif ".dat" in open_path:
                        self.update_progress("Loading nuclei centroids from text file...")
                        self.remove_centroids()
                        self.centroids, self.dot_pos_list_stk = load_nuclei_list(open_path)
                        self.update_progress("Determining Closest Contour Positions...")
                        # self.closest_contour_pixels = seg.get_closest_contour_px(self.contour,self.centroids)
                        self.draw_img()
                        self.update_progress("Nuclei centroids loaded!")

                    else:
                        QtGui.QMessageBox.about(self,"Title","Please select a valid <<.tif>> or <<.dat>> file to load nuclei centroids!")

            else:
                QtGui.QMessageBox.about(self,"Title","Please load or find the spheroid contour first!")

        else:
            return

    def save_nuclei(self, save_path=False):

        if not self.centroids is None:

            self.get_centroids_from_dots()

            if not save_path:
                if not self.im_path is None:
                    path_hint = self.im_path.replace(".tif", "_nuclei_centroids.tif")
                else:
                    path_hint = "./nuclei_centroids.tif"

                save_path = QtGui.QFileDialog.getSaveFileName(self, QtCore.QString('Select save path for nuclei centroid image'),
                                                              path_hint)

            if not save_path is None:
                save_nuclei_centroids(self.im.stk_array,self.centroids,save_path)

            self.update_progress("Nuclei Centroids Saved!")

        else:
            QtGui.QMessageBox.about(self,"Title","No nuclei centroids to save!")


    def set_disp_params(self,h,w):

        #sets parameters for displaying images
        self.ui.widget_img.set_img_window_params(h,w)

    def clear_data(self):

        if self.ask_permission("Do you really want to clear the current centroid data?"):
            self.remove_centroids()

    def remove_centroids(self):

        self.dot_pos_list_stk = [[] for _ in range(self.im.no_slices)]
        self.ui.widget_img.delete_items()
        self.draw_img()
        self.centroids = None

    def next_slice(self):
        if self.current_slice + 1 <= self.im.no_slices:
            self.current_slice += 1
        self.draw_img()

    def prev_slice(self):
        if self.current_slice - 1 > 0:
            self.current_slice -= 1
        self.draw_img()

    def go_to_slice(self):
        self.current_slice = int(self.ui.text_current_slice.text())
        self.draw_img()

    def draw_img(self):

        if not self.im is None:

            #check which boxes are turned on to display data
            if self.ui.box_disp_img.checkState()==2:
                img_data = self.im.stk_array[self.current_slice-1]
            else:
                img_data = np.zeros(shape=(self.im.stk_array[self.current_slice-1]).shape)

            if self.ui.box_disp_sph.checkState()==2:
                sph_data = 255*self.contour[self.current_slice-1]
            else:
                sph_data = np.zeros(shape=(self.contour[self.current_slice-1]).shape)

            #update dot position list (for current frame)
            self.ui.widget_img.dot_pos_list = self.dot_pos_list_stk[self.current_slice-1]
            self.ui.widget_img.disp_img(img_data,self.im.im_size[1],self.im.im_size[2],
                                        blue_data=sph_data)

            #make the dots invisible if necessary
            if self.ui.box_disp_centroids.checkState()==2:
                pass
            else:
                self.ui.widget_img.redraw_dots(visible=False)

            self.update_slice_text()

    def update_slice_text(self):
        self.ui.text_current_slice.setText(str(self.current_slice))
        self.ui.text_current_slice.update()
        self.ui.text_no_slices.setText(str(self.im.no_slices))
        self.ui.text_no_slices.update()

    def update_dot_positions(self,dot_pos_list):
        # print dot_pos_list
        self.dot_pos_list_stk[self.current_slice-1] = dot_pos_list
        self.get_centroids_from_dots()

    def get_centroids_from_dots(self):

        self.centroids = []
        for i in range(len(self.dot_pos_list_stk)):
            for j in range(len(self.dot_pos_list_stk[i])):
                self.centroids.append([self.dot_pos_list_stk[i][j][0],self.dot_pos_list_stk[i][j][1],i])

    def get_current_settings(self):
        """Gets parameters for segmentation from the ui"""

        #checks if image should be scaled
        self.scaling = self.ui.box_scaling.currentText()
        if self.scaling=="None":
            self.scaling=False

        #checks output toggle
        if self.ui.box_write_int_images.checkState()==2:
            self.output_toggle = True
        else:
            self.output_toggle = False

        #gets spheroid parameters from gui
        if self.ui.box_use_stk_hist_sph.checkState()==2:
            self.stk_hist_sph = True
        else:
            self.stk_hist_sph = False
        self.sigma_sph = int(self.ui.box_sigma_sph.value())
        self.thresh_type_sph = str(self.ui.box_thresh_type_sph.currentText())
        self.thresh_factor_sph = self.ui.box_thresh_factor_sph.value()
        self.erosion_factor_sph = int(self.ui.box_erosion_fact_sph.value())
        self.smoothing_factor_sph = int(self.ui.box_smoothing_fact_sph.value())

        #gets nuclei parameters from gui
        if self.ui.box_use_stk_hist_nuclei.checkState()==2:
            self.stk_hist_nuclei = True
        else:
            self.stk_hist_nuclei = False
        self.sigma_nuclei = int(self.ui.box_sigma_nuclei.value())
        self.thresh_type_nuclei = str(self.ui.box_thresh_type_nuclei.currentText())
        self.thresh_factor_nuclei = self.ui.box_thresh_factor_nuclei.value()
        self.min_size_nuclei = int(self.ui.box_min_size_nuclei.value())

    def find_spheroid(self):

        if not self.im is None:

            self.get_current_settings()
            self.update_progress("Finding Spheroid Contour...")
            self.contour,self.mask = seg.get_spheroid_contour(self.im.stk_array, self.im_path,
                                                              stk_hist=self.stk_hist_sph,
                                                              sigma=self.sigma_sph,
                                                              thresh_type=self.thresh_type_sph,
                                                              thresh_factor=self.thresh_factor_sph,
                                                              erosion_factor=self.erosion_factor_sph,
                                                              smoothing_factor=self.smoothing_factor_sph,
                                                              output_toggle=self.output_toggle)
            self.update_progress("Spheroid Contour Found!")
            self.draw_img()

        else:
            QtGui.QMessageBox.about(self,"Title","Please load an image first!")


    def find_nuclei(self):

        self.get_current_settings()

        if not self.contour is None and not np.max(self.contour)==0:

            self.update_progress("Segmenting Nuclei...")
            self.nuclei = seg.get_nuclei(self.im.stk_array, self.mask, self.im_path,
                                         stk_hist=self.stk_hist_nuclei,
                                         sigma=self.sigma_nuclei,
                                         thresh_type=self.thresh_type_nuclei,
                                         thresh_factor=self.thresh_factor_nuclei,
                                         min_size=self.min_size_nuclei,
                                         output_toggle=self.output_toggle)

            self.update_progress("Determining Centroid Positions...")
            self.centroids = seg.get_centroids(self.im.stk_array,self.nuclei,self.im_path,output_toggle=self.output_toggle)

            #displays the centroids (be careful of the change in indexing between the centroids array and the dot list!)
            self.dot_pos_list_stk = [[] for _ in range(self.im.no_slices)]
            for coords in self.centroids:
                (self.dot_pos_list_stk[int(coords[2])]).append([int(coords[0]),int(coords[1])])
            self.draw_img()
            self.update_progress("Nuclei Centroids Found!")

        else:
            QtGui.QMessageBox.about(self,"Title","Please segment the spheroid first!")

    def plot_get_stats(self, save_path=False):

        if not self.im_path is None and not self.contour is None and not self.centroids is None:

            #gets save_path if necessary
            if not save_path:
                if not self.im_path is None:
                    path_hint = self.im_path.replace(".tif", "_plot.png")
                if self.im_path is None:
                    path_hint = "./_plot.png"

                save_path = str(QtGui.QFileDialog.getSaveFileName(self, QtCore.QString('Select settings save path'),
                                                                  path_hint))

            if save_path:

                self.px_size_xy = float(self.ui.text_px_size_xy.text())
                self.px_size_z = float(self.ui.text_px_size_z.text())
                im_name = self.im_path.split("/")[-1][:-4]

                # gets closest contour pixels
                self.update_progress("Determining Closest Contour Positions...")
                self.closest_contour_pixels = seg.get_closest_contour_px(self.contour,self.centroids,self.px_size_xy,self.px_size_z)
                self.reset_progress()

                # print "Saving and Plotting!"
                self.update_progress("Generating Invasion Plots and Statistics...")
                seg.plot_invasion(self.contour,self.centroids,self.closest_contour_pixels,
                                  save_path,im_name,self.px_size_xy,self.px_size_z)
                self.update_progress("Invasion Plots and Statistics Generated!")

        else:
            QtGui.QMessageBox.about(self,"Warning!!","Please load the image, segment the spheroid and find nuclei first!")

    def save_settings(self, save_path=False):

        self.get_current_settings()

        settings_list = ["stk_hist_sph",
                         "thresh_type_sph",
                         "thresh_factor_sph",
                         "erosion_factor_sph",
                         "smoothing_factor_sph",
                         "sigma_sph",
                         "stk_hist_nuclei",
                         "thresh_type_nuclei",
                         "thresh_factor_nuclei",
                         "min_size_nuclei",
                         "sigma_nuclei",
                         "px_size_xy",
                         "px_size_z"]

        values_list = [eval("self.%s"%name) for name in settings_list]

        #gets save path
        if not save_path:
            if not self.im_path is None:
                save_path = QtGui.QFileDialog.getSaveFileName(self, QtCore.QString('Select settings save path'),
                                                              self.im_path.replace(".tif", "_settings.dat"))
            else:
                save_path = QtGui.QFileDialog.getSaveFileName(self, QtCore.QString('Select settings save path'),
                                                              "./settings.dat")

        if save_path:
            save_data_array([settings_list,values_list],save_path)

    def load_settings(self, open_path=False):

        if not open_path:
            #gets settings from file
            if not self.im_path is None:
                open_path = QtGui.QFileDialog.getOpenFileName(self, 'Select settings file',
                                                                  self.im_path.replace(".tif", "_settings.dat"))
            else:
                open_path = QtGui.QFileDialog.getOpenFileName(self, 'Select settings file','./settings.dat')

            open_path = str(open_path)

        if open_path:

            settings_data = read_file(open_path)
            settings = {}
            for i in range(len(settings_data[0])):
                settings[settings_data[0][i]] = settings_data[1][i]

            #populate gui with settings
            if settings["stk_hist_sph"]=="True":
                self.ui.box_use_stk_hist_sph.setCheckState(QtCore.Qt.Checked)
            else:
                self.ui.box_use_stk_hist_sph.setCheckState(QtCore.Qt.Unchecked)

            if settings["thresh_type_sph"]=="Moments":
                self.ui.box_thresh_type_sph.setCurrentIndex(0)
            elif settings["thresh_type_sph"]=="Otsu":
                self.ui.box_thresh_type_sph.setCurrentIndex(1)
            else:
                QtGui.QMessageBox.about(self,"Title","Unknown threshold type <<%s>> for spheroid semgentation!"%self.ui.box_thresh_type_sph)

            self.ui.box_thresh_factor_sph.setValue(float(settings["thresh_factor_sph"]))
            self.ui.box_erosion_fact_sph.setValue(int(settings["erosion_factor_sph"]))
            self.ui.box_smoothing_fact_sph.setValue(int(settings["smoothing_factor_sph"]))
            self.ui.box_sigma_sph.setValue(int(settings["sigma_sph"]))

            if settings["stk_hist_nuclei"]=="True":
                self.ui.box_use_stk_hist_nuclei.setCheckState(QtCore.Qt.Checked)
            else:
                self.ui.box_use_stk_hist_nuclei.setCheckState(QtCore.Qt.Unchecked)

            if settings["thresh_type_nuclei"]=="Moments":
                self.ui.box_thresh_type_nuclei.setCurrentIndex(0)
            elif settings["thresh_type_nuclei"]=="Otsu":
                self.ui.box_thresh_type_nuclei.setCurrentIndex(1)
            else:
                QtGui.QMessageBox.about(self,"Title","Unknown threshold type <<%s>> for nuclei semgentation!"%self.ui.box_thresh_type_nuclei)

            self.ui.box_thresh_factor_nuclei.setValue(float(settings["thresh_factor_nuclei"]))
            self.ui.box_min_size_nuclei.setValue(int(settings["min_size_nuclei"]))
            self.ui.box_sigma_nuclei.setValue(int(settings["sigma_nuclei"]))

            self.ui.text_px_size_xy.setText(settings["px_size_xy"])
            self.ui.text_px_size_z.setText(settings["px_size_z"])

            self.get_current_settings()

    def restore_default_settings(self):

        if self.ask_permission("Are you sure you want to restore the default settings?"):
            self.load_default_settings()

    def load_default_settings(self):

        #populate gui with settings
        self.ui.box_scaling.setCurrentIndex(0)
        self.ui.box_use_stk_hist_sph.setCheckState(QtCore.Qt.Checked)
        self.ui.box_thresh_type_sph.setCurrentIndex(0)
        self.ui.box_thresh_factor_sph.setValue(1.0)
        self.ui.box_erosion_fact_sph.setValue(5)
        self.ui.box_smoothing_fact_sph.setValue(40)
        self.ui.box_sigma_sph.setValue(2)
        self.ui.box_use_stk_hist_nuclei.setCheckState(QtCore.Qt.Checked)
        self.ui.box_thresh_type_nuclei.setCurrentIndex(0)
        self.ui.box_thresh_factor_nuclei.setValue(1.0)
        self.ui.box_min_size_nuclei.setValue(250)
        self.ui.box_sigma_nuclei.setValue(2)
        self.ui.text_px_size_xy.setText("1.0")
        self.ui.text_px_size_z.setText("1.0")
        self.ui.box_disp_img.setCheckState(QtCore.Qt.Checked)
        self.ui.box_disp_sph.setCheckState(QtCore.Qt.Checked)
        self.ui.box_disp_centroids.setCheckState(QtCore.Qt.Checked)

        self.get_current_settings()

    def save_everything(self):

        if not self.im_path is None and not self.contour is None and not self.centroids is None:

            answer = self.ask_permission("This will automatically overwrite any contour/spheroid/plot files! Proceed?")
            if answer:

                #gets the base save path
                save_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Select base file name', self.im_path))

                #saves settings, contour, centroids, invasion plot and stats
                self.save_settings(save_path=save_path.replace(".tif","_settings.dat"))
                self.save_spheroid(save_path=save_path.replace(".tif","_spheroid_contour.tif"))
                self.save_nuclei(save_path=save_path.replace(".tif","_nuclei_centroids.tif"))
                self.plot_get_stats(save_path=save_path.replace(".tif","_plot.png"))

                self.update_progress("Everything saved!")

        else:
            QtGui.QMessageBox.about(self,"Warning!!","Please load the image, segment the spheroid and find nuclei first!")

    def load_everything(self):

        if self.ask_permission("Do you really want to clear the current image and data?"):

            #gets the open path for the new images/files
            if not self.im_path is None:
                path_hint = self.im_path
            else:
                path_hint = "./.tif"
            self.im_path = str(QtGui.QFileDialog.getOpenFileName(self, 'Select a new stack to open (associated files must be in same directory)',
                                                                 path_hint))

            self.load_settings(open_path=self.im_path.replace(".tif","_settings.dat"))
            self.load_stk(open_path=self.im_path)
            self.load_spheroid(open_path=self.im_path.replace(".tif","_spheroid_contour.tif"))
            self.load_nuclei(open_path=self.im_path.replace(".tif","_nuclei_centroids.dat"),ask=False)
            self.update_progress("Everything loaded!")

        else:
            return

    def update_progress(self,text):

        self.ui.label_progress.setText(text)
        self.ui.label_progress.repaint()
        QtGui.QApplication.processEvents()

    def reset_progress(self):

        self.ui.label_progress.setText("Invasion Counter v1.2")
        self.ui.label_progress.repaint()
        QtGui.QApplication.processEvents()

    def ask_permission(self,question):

        answer = QtGui.QMessageBox.question(self,'Warning!!', question, QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if answer==QtGui.QMessageBox.Yes:
            return True
        else:
            return False

def main():

    app = QtGui.QApplication(sys.argv)
    app.processEvents()
    simple = Simple()
    simple.show()
    simple.raise_()
    #print "simple dir", dir(simple)
    sys.exit(app.exec_())


if __name__ == '__main__':

    main()
