# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'invasion_counter_gui_v1_2.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_TestMainWindow(object):
    def setupUi(self, TestMainWindow):
        TestMainWindow.setObjectName(_fromUtf8("TestMainWindow"))
        TestMainWindow.resize(1087, 711)
        self.widget_img = MyGraphicsView(TestMainWindow)
        self.widget_img.setGeometry(QtCore.QRect(260, 10, 651, 651))
        self.widget_img.setObjectName(_fromUtf8("widget_img"))
        self.button_clear_data = QtGui.QPushButton(TestMainWindow)
        self.button_clear_data.setGeometry(QtCore.QRect(10, 650, 91, 32))
        self.button_clear_data.setObjectName(_fromUtf8("button_clear_data"))
        self.button_load_stk = QtGui.QPushButton(TestMainWindow)
        self.button_load_stk.setGeometry(QtCore.QRect(10, 10, 121, 32))
        self.button_load_stk.setObjectName(_fromUtf8("button_load_stk"))
        self.button_previous = QtGui.QPushButton(TestMainWindow)
        self.button_previous.setGeometry(QtCore.QRect(350, 670, 131, 32))
        self.button_previous.setObjectName(_fromUtf8("button_previous"))
        self.label_5 = QtGui.QLabel(TestMainWindow)
        self.label_5.setGeometry(QtCore.QRect(560, 670, 21, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.button_go = QtGui.QPushButton(TestMainWindow)
        self.button_go.setGeometry(QtCore.QRect(780, 670, 51, 32))
        self.button_go.setObjectName(_fromUtf8("button_go"))
        self.text_current_slice = QtGui.QLineEdit(TestMainWindow)
        self.text_current_slice.setGeometry(QtCore.QRect(490, 670, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.text_current_slice.setFont(font)
        self.text_current_slice.setMaxLength(3)
        self.text_current_slice.setObjectName(_fromUtf8("text_current_slice"))
        self.button_next = QtGui.QPushButton(TestMainWindow)
        self.button_next.setGeometry(QtCore.QRect(650, 670, 131, 32))
        self.button_next.setObjectName(_fromUtf8("button_next"))
        self.text_no_slices = QtGui.QLineEdit(TestMainWindow)
        self.text_no_slices.setGeometry(QtCore.QRect(580, 670, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.text_no_slices.setFont(font)
        self.text_no_slices.setMaxLength(3)
        self.text_no_slices.setReadOnly(True)
        self.text_no_slices.setObjectName(_fromUtf8("text_no_slices"))
        self.groupBox = QtGui.QGroupBox(TestMainWindow)
        self.groupBox.setGeometry(QtCore.QRect(30, 350, 221, 201))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.button_find_nuclei = QtGui.QPushButton(self.groupBox)
        self.button_find_nuclei.setGeometry(QtCore.QRect(0, 20, 61, 32))
        self.button_find_nuclei.setObjectName(_fromUtf8("button_find_nuclei"))
        self.box_use_stk_hist_nuclei = QtGui.QCheckBox(self.groupBox)
        self.box_use_stk_hist_nuclei.setGeometry(QtCore.QRect(10, 50, 161, 20))
        self.box_use_stk_hist_nuclei.setChecked(True)
        self.box_use_stk_hist_nuclei.setObjectName(_fromUtf8("box_use_stk_hist_nuclei"))
        self.label_8 = QtGui.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(110, 110, 91, 31))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.box_thresh_type_nuclei = QtGui.QComboBox(self.groupBox)
        self.box_thresh_type_nuclei.setGeometry(QtCore.QRect(10, 110, 91, 26))
        self.box_thresh_type_nuclei.setObjectName(_fromUtf8("box_thresh_type_nuclei"))
        self.box_thresh_type_nuclei.addItem(_fromUtf8(""))
        self.box_thresh_type_nuclei.addItem(_fromUtf8(""))
        self.label_9 = QtGui.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(90, 140, 111, 31))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.box_thresh_factor_nuclei = QtGui.QDoubleSpinBox(self.groupBox)
        self.box_thresh_factor_nuclei.setGeometry(QtCore.QRect(20, 140, 62, 24))
        self.box_thresh_factor_nuclei.setReadOnly(False)
        self.box_thresh_factor_nuclei.setDecimals(1)
        self.box_thresh_factor_nuclei.setMaximum(5.0)
        self.box_thresh_factor_nuclei.setSingleStep(0.1)
        self.box_thresh_factor_nuclei.setProperty("value", 1.0)
        self.box_thresh_factor_nuclei.setObjectName(_fromUtf8("box_thresh_factor_nuclei"))
        self.label_10 = QtGui.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(90, 170, 111, 31))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.box_min_size_nuclei = QtGui.QSpinBox(self.groupBox)
        self.box_min_size_nuclei.setGeometry(QtCore.QRect(20, 170, 61, 24))
        self.box_min_size_nuclei.setMaximum(1000)
        self.box_min_size_nuclei.setSingleStep(10)
        self.box_min_size_nuclei.setProperty("value", 250)
        self.box_min_size_nuclei.setObjectName(_fromUtf8("box_min_size_nuclei"))
        self.box_sigma_nuclei = QtGui.QSpinBox(self.groupBox)
        self.box_sigma_nuclei.setGeometry(QtCore.QRect(20, 80, 61, 24))
        self.box_sigma_nuclei.setMaximum(20)
        self.box_sigma_nuclei.setProperty("value", 2)
        self.box_sigma_nuclei.setObjectName(_fromUtf8("box_sigma_nuclei"))
        self.label_11 = QtGui.QLabel(self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(90, 80, 91, 31))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.button_load_nuclei = QtGui.QPushButton(self.groupBox)
        self.button_load_nuclei.setGeometry(QtCore.QRect(70, 20, 61, 32))
        self.button_load_nuclei.setObjectName(_fromUtf8("button_load_nuclei"))
        self.button_save_nuclei = QtGui.QPushButton(self.groupBox)
        self.button_save_nuclei.setGeometry(QtCore.QRect(140, 20, 61, 32))
        self.button_save_nuclei.setObjectName(_fromUtf8("button_save_nuclei"))
        self.groupBox_2 = QtGui.QGroupBox(TestMainWindow)
        self.groupBox_2.setGeometry(QtCore.QRect(950, 60, 121, 151))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.text_px_size_xy = QtGui.QLineEdit(self.groupBox_2)
        self.text_px_size_xy.setGeometry(QtCore.QRect(40, 30, 61, 21))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setItalic(True)
        self.text_px_size_xy.setFont(font)
        self.text_px_size_xy.setMaxLength(8)
        self.text_px_size_xy.setObjectName(_fromUtf8("text_px_size_xy"))
        self.label = QtGui.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(10, 30, 21, 16))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setItalic(False)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 70, 16, 16))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setItalic(False)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.text_px_size_z = QtGui.QLineEdit(self.groupBox_2)
        self.text_px_size_z.setGeometry(QtCore.QRect(40, 70, 61, 21))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setItalic(True)
        self.text_px_size_z.setFont(font)
        self.text_px_size_z.setMaxLength(7)
        self.text_px_size_z.setObjectName(_fromUtf8("text_px_size_z"))
        self.text_px_size_xy.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.text_px_size_z.raise_()
        self.button_previous.raise_()
        self.button_go.raise_()
        self.text_current_slice.raise_()
        self.label_5.raise_()
        self.button_next.raise_()
        self.text_no_slices.raise_()
        self.groupBox_spheroid = QtGui.QGroupBox(TestMainWindow)
        self.groupBox_spheroid.setGeometry(QtCore.QRect(30, 110, 221, 231))
        self.groupBox_spheroid.setObjectName(_fromUtf8("groupBox_spheroid"))
        self.button_find_spheroid = QtGui.QPushButton(self.groupBox_spheroid)
        self.button_find_spheroid.setGeometry(QtCore.QRect(0, 20, 61, 32))
        self.button_find_spheroid.setObjectName(_fromUtf8("button_find_spheroid"))
        self.box_use_stk_hist_sph = QtGui.QCheckBox(self.groupBox_spheroid)
        self.box_use_stk_hist_sph.setGeometry(QtCore.QRect(10, 50, 161, 20))
        self.box_use_stk_hist_sph.setChecked(True)
        self.box_use_stk_hist_sph.setObjectName(_fromUtf8("box_use_stk_hist_sph"))
        self.box_thresh_type_sph = QtGui.QComboBox(self.groupBox_spheroid)
        self.box_thresh_type_sph.setGeometry(QtCore.QRect(10, 110, 91, 26))
        self.box_thresh_type_sph.setObjectName(_fromUtf8("box_thresh_type_sph"))
        self.box_thresh_type_sph.addItem(_fromUtf8(""))
        self.box_thresh_type_sph.addItem(_fromUtf8(""))
        self.label_3 = QtGui.QLabel(self.groupBox_spheroid)
        self.label_3.setGeometry(QtCore.QRect(110, 110, 91, 31))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.box_thresh_factor_sph = QtGui.QDoubleSpinBox(self.groupBox_spheroid)
        self.box_thresh_factor_sph.setGeometry(QtCore.QRect(20, 140, 62, 24))
        self.box_thresh_factor_sph.setReadOnly(False)
        self.box_thresh_factor_sph.setDecimals(1)
        self.box_thresh_factor_sph.setMaximum(5.0)
        self.box_thresh_factor_sph.setSingleStep(0.1)
        self.box_thresh_factor_sph.setProperty("value", 1.0)
        self.box_thresh_factor_sph.setObjectName(_fromUtf8("box_thresh_factor_sph"))
        self.label_4 = QtGui.QLabel(self.groupBox_spheroid)
        self.label_4.setGeometry(QtCore.QRect(90, 140, 111, 31))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.box_erosion_fact_sph = QtGui.QSpinBox(self.groupBox_spheroid)
        self.box_erosion_fact_sph.setGeometry(QtCore.QRect(20, 170, 61, 24))
        self.box_erosion_fact_sph.setMaximum(100)
        self.box_erosion_fact_sph.setProperty("value", 5)
        self.box_erosion_fact_sph.setObjectName(_fromUtf8("box_erosion_fact_sph"))
        self.label_6 = QtGui.QLabel(self.groupBox_spheroid)
        self.label_6.setGeometry(QtCore.QRect(90, 170, 111, 31))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.box_smoothing_fact_sph = QtGui.QSpinBox(self.groupBox_spheroid)
        self.box_smoothing_fact_sph.setGeometry(QtCore.QRect(20, 200, 61, 24))
        self.box_smoothing_fact_sph.setMaximum(100)
        self.box_smoothing_fact_sph.setProperty("value", 40)
        self.box_smoothing_fact_sph.setObjectName(_fromUtf8("box_smoothing_fact_sph"))
        self.label_7 = QtGui.QLabel(self.groupBox_spheroid)
        self.label_7.setGeometry(QtCore.QRect(90, 200, 111, 31))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.box_sigma_sph = QtGui.QSpinBox(self.groupBox_spheroid)
        self.box_sigma_sph.setGeometry(QtCore.QRect(20, 80, 61, 24))
        self.box_sigma_sph.setMaximum(20)
        self.box_sigma_sph.setProperty("value", 2)
        self.box_sigma_sph.setObjectName(_fromUtf8("box_sigma_sph"))
        self.label_12 = QtGui.QLabel(self.groupBox_spheroid)
        self.label_12.setGeometry(QtCore.QRect(90, 80, 91, 31))
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.button_load_spheroid = QtGui.QPushButton(self.groupBox_spheroid)
        self.button_load_spheroid.setGeometry(QtCore.QRect(70, 20, 61, 32))
        self.button_load_spheroid.setObjectName(_fromUtf8("button_load_spheroid"))
        self.button_save_spheroid = QtGui.QPushButton(self.groupBox_spheroid)
        self.button_save_spheroid.setGeometry(QtCore.QRect(140, 20, 61, 32))
        self.button_save_spheroid.setObjectName(_fromUtf8("button_save_spheroid"))
        self.button_find_spheroid.raise_()
        self.box_use_stk_hist_sph.raise_()
        self.box_thresh_type_sph.raise_()
        self.label_3.raise_()
        self.box_thresh_factor_sph.raise_()
        self.label_4.raise_()
        self.box_erosion_fact_sph.raise_()
        self.label_6.raise_()
        self.box_smoothing_fact_sph.raise_()
        self.label_7.raise_()
        self.box_sigma_sph.raise_()
        self.label_12.raise_()
        self.button_load_spheroid.raise_()
        self.button_load_stk.raise_()
        self.button_save_spheroid.raise_()
        self.groupBox_4 = QtGui.QGroupBox(TestMainWindow)
        self.groupBox_4.setGeometry(QtCore.QRect(950, 210, 121, 161))
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.box_disp_img = QtGui.QCheckBox(self.groupBox_4)
        self.box_disp_img.setGeometry(QtCore.QRect(20, 30, 61, 20))
        self.box_disp_img.setChecked(True)
        self.box_disp_img.setObjectName(_fromUtf8("box_disp_img"))
        self.box_disp_sph = QtGui.QCheckBox(self.groupBox_4)
        self.box_disp_sph.setGeometry(QtCore.QRect(20, 60, 121, 20))
        self.box_disp_sph.setChecked(True)
        self.box_disp_sph.setObjectName(_fromUtf8("box_disp_sph"))
        self.box_disp_centroids = QtGui.QCheckBox(self.groupBox_4)
        self.box_disp_centroids.setGeometry(QtCore.QRect(20, 90, 81, 20))
        self.box_disp_centroids.setChecked(True)
        self.box_disp_centroids.setObjectName(_fromUtf8("box_disp_centroids"))
        self.button_redraw = QtGui.QPushButton(self.groupBox_4)
        self.button_redraw.setGeometry(QtCore.QRect(20, 120, 81, 32))
        self.button_redraw.setObjectName(_fromUtf8("button_redraw"))
        self.button_load_settings = QtGui.QPushButton(TestMainWindow)
        self.button_load_settings.setGeometry(QtCore.QRect(140, 10, 121, 32))
        self.button_load_settings.setObjectName(_fromUtf8("button_load_settings"))
        self.button_save_settings = QtGui.QPushButton(TestMainWindow)
        self.button_save_settings.setGeometry(QtCore.QRect(140, 40, 121, 32))
        self.button_save_settings.setObjectName(_fromUtf8("button_save_settings"))
        self.button_plot_get_stats = QtGui.QPushButton(TestMainWindow)
        self.button_plot_get_stats.setGeometry(QtCore.QRect(60, 80, 171, 32))
        self.button_plot_get_stats.setObjectName(_fromUtf8("button_plot_get_stats"))
        self.box_write_int_images = QtGui.QCheckBox(TestMainWindow)
        self.box_write_int_images.setGeometry(QtCore.QRect(20, 620, 181, 31))
        self.box_write_int_images.setChecked(False)
        self.box_write_int_images.setObjectName(_fromUtf8("box_write_int_images"))
        self.label_progress = QtGui.QLabel(TestMainWindow)
        self.label_progress.setGeometry(QtCore.QRect(20, 690, 371, 16))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setItalic(True)
        self.label_progress.setFont(font)
        self.label_progress.setObjectName(_fromUtf8("label_progress"))
        self.box_scaling = QtGui.QComboBox(TestMainWindow)
        self.box_scaling.setGeometry(QtCore.QRect(20, 40, 71, 26))
        self.box_scaling.setMaxVisibleItems(3)
        self.box_scaling.setObjectName(_fromUtf8("box_scaling"))
        self.box_scaling.addItem(_fromUtf8(""))
        self.box_scaling.addItem(_fromUtf8(""))
        self.box_scaling.addItem(_fromUtf8(""))
        self.label_13 = QtGui.QLabel(TestMainWindow)
        self.label_13.setGeometry(QtCore.QRect(90, 40, 51, 31))
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.button_default_settings = QtGui.QPushButton(TestMainWindow)
        self.button_default_settings.setGeometry(QtCore.QRect(100, 650, 141, 32))
        self.button_default_settings.setObjectName(_fromUtf8("button_default_settings"))
        self.button_save_everything = QtGui.QPushButton(TestMainWindow)
        self.button_save_everything.setGeometry(QtCore.QRect(60, 560, 141, 31))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setItalic(True)
        self.button_save_everything.setFont(font)
        self.button_save_everything.setObjectName(_fromUtf8("button_save_everything"))
        self.button_load_everything = QtGui.QPushButton(TestMainWindow)
        self.button_load_everything.setGeometry(QtCore.QRect(60, 590, 141, 31))
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setItalic(True)
        self.button_load_everything.setFont(font)
        self.button_load_everything.setObjectName(_fromUtf8("button_load_everything"))

        self.retranslateUi(TestMainWindow)
        self.box_scaling.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(TestMainWindow)

    def retranslateUi(self, TestMainWindow):
        TestMainWindow.setWindowTitle(_translate("TestMainWindow", "Invasion Counter", None))
        self.button_clear_data.setText(_translate("TestMainWindow", "Clear Data", None))
        self.button_load_stk.setText(_translate("TestMainWindow", "Load Stack", None))
        self.button_previous.setText(_translate("TestMainWindow", "<- Previous Slice", None))
        self.label_5.setText(_translate("TestMainWindow", "/", None))
        self.button_go.setText(_translate("TestMainWindow", "Go!", None))
        self.button_next.setText(_translate("TestMainWindow", "Next Slice ->", None))
        self.groupBox.setTitle(_translate("TestMainWindow", "Nuclei Centroids", None))
        self.button_find_nuclei.setText(_translate("TestMainWindow", "Find", None))
        self.box_use_stk_hist_nuclei.setText(_translate("TestMainWindow", "Use stack histogram", None))
        self.label_8.setText(_translate("TestMainWindow", "Threshold Type", None))
        self.box_thresh_type_nuclei.setItemText(0, _translate("TestMainWindow", "Moments", None))
        self.box_thresh_type_nuclei.setItemText(1, _translate("TestMainWindow", "Otsu", None))
        self.label_9.setText(_translate("TestMainWindow", "Threshold Factor", None))
        self.label_10.setText(_translate("TestMainWindow", "Minimum Size", None))
        self.label_11.setText(_translate("TestMainWindow", "Sigma", None))
        self.button_load_nuclei.setText(_translate("TestMainWindow", "Load", None))
        self.button_save_nuclei.setText(_translate("TestMainWindow", "Save", None))
        self.groupBox_2.setTitle(_translate("TestMainWindow", "Pixel Sizes (μm)", None))
        self.text_px_size_xy.setText(_translate("TestMainWindow", "1.0", None))
        self.label.setText(_translate("TestMainWindow", "xy", None))
        self.label_2.setText(_translate("TestMainWindow", "z", None))
        self.text_px_size_z.setText(_translate("TestMainWindow", "1.0", None))
        self.groupBox_spheroid.setTitle(_translate("TestMainWindow", "Spheroid Contour", None))
        self.button_find_spheroid.setText(_translate("TestMainWindow", "Find", None))
        self.box_use_stk_hist_sph.setText(_translate("TestMainWindow", "Use stack histogram", None))
        self.box_thresh_type_sph.setItemText(0, _translate("TestMainWindow", "Moments", None))
        self.box_thresh_type_sph.setItemText(1, _translate("TestMainWindow", "Otsu", None))
        self.label_3.setText(_translate("TestMainWindow", "Threshold Type", None))
        self.label_4.setText(_translate("TestMainWindow", "Threshold Factor", None))
        self.label_6.setText(_translate("TestMainWindow", "Erosion Factor", None))
        self.label_7.setText(_translate("TestMainWindow", "Smoothing Factor", None))
        self.label_12.setText(_translate("TestMainWindow", "Sigma", None))
        self.button_load_spheroid.setText(_translate("TestMainWindow", "Load", None))
        self.button_save_spheroid.setText(_translate("TestMainWindow", "Save", None))
        self.groupBox_4.setTitle(_translate("TestMainWindow", "Display Options", None))
        self.box_disp_img.setText(_translate("TestMainWindow", "Image", None))
        self.box_disp_sph.setText(_translate("TestMainWindow", "Contour", None))
        self.box_disp_centroids.setText(_translate("TestMainWindow", "Centroids", None))
        self.button_redraw.setText(_translate("TestMainWindow", "Redraw", None))
        self.button_load_settings.setText(_translate("TestMainWindow", "Load Settings", None))
        self.button_save_settings.setText(_translate("TestMainWindow", "Save Settings", None))
        self.button_plot_get_stats.setText(_translate("TestMainWindow", "Plot / Get Statistics", None))
        self.box_write_int_images.setText(_translate("TestMainWindow", "Write Intermediate Images", None))
        self.label_progress.setText(_translate("TestMainWindow", "Invasion Counter v1.2", None))
        self.box_scaling.setItemText(0, _translate("TestMainWindow", "None", None))
        self.box_scaling.setItemText(1, _translate("TestMainWindow", "Stack", None))
        self.box_scaling.setItemText(2, _translate("TestMainWindow", "Slices", None))
        self.label_13.setText(_translate("TestMainWindow", "Scaling", None))
        self.button_default_settings.setText(_translate("TestMainWindow", "Default Settings", None))
        self.button_save_everything.setText(_translate("TestMainWindow", "Save Everything", None))
        self.button_load_everything.setText(_translate("TestMainWindow", "Load Everything", None))

from MyQgraphics_widget_v3 import MyGraphicsView