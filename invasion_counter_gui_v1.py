# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'invasion_counter_gui_v1.ui'
#
# Created: Fri Jul  3 17:33:39 2015
#      by: PyQt4 UI code generator 4.11.3
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
        TestMainWindow.resize(582, 512)
        self.widget_img = MyGraphicsView(TestMainWindow)
        self.widget_img.setGeometry(QtCore.QRect(10, 10, 420, 420))
        self.widget_img.setObjectName(_fromUtf8("widget_img"))
        self.button_clear_init = QtGui.QPushButton(TestMainWindow)
        self.button_clear_init.setGeometry(QtCore.QRect(440, 120, 131, 32))
        self.button_clear_init.setObjectName(_fromUtf8("button_clear_init"))
        self.button_load = QtGui.QPushButton(TestMainWindow)
        self.button_load.setGeometry(QtCore.QRect(440, 10, 131, 32))
        self.button_load.setObjectName(_fromUtf8("button_load"))
        self.box_scale = QtGui.QCheckBox(TestMainWindow)
        self.box_scale.setGeometry(QtCore.QRect(460, 80, 111, 20))
        self.box_scale.setChecked(True)
        self.box_scale.setObjectName(_fromUtf8("box_scale"))
        self.button_load_stk = QtGui.QPushButton(TestMainWindow)
        self.button_load_stk.setGeometry(QtCore.QRect(440, 40, 131, 32))
        self.button_load_stk.setObjectName(_fromUtf8("button_load_stk"))
        self.button_previous = QtGui.QPushButton(TestMainWindow)
        self.button_previous.setGeometry(QtCore.QRect(0, 440, 151, 32))
        self.button_previous.setObjectName(_fromUtf8("button_previous"))
        self.label_5 = QtGui.QLabel(TestMainWindow)
        self.label_5.setGeometry(QtCore.QRect(230, 440, 21, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.button_go = QtGui.QPushButton(TestMainWindow)
        self.button_go.setGeometry(QtCore.QRect(210, 480, 51, 32))
        self.button_go.setObjectName(_fromUtf8("button_go"))
        self.text_current_slice = QtGui.QLineEdit(TestMainWindow)
        self.text_current_slice.setGeometry(QtCore.QRect(160, 440, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.text_current_slice.setFont(font)
        self.text_current_slice.setMaxLength(3)
        self.text_current_slice.setObjectName(_fromUtf8("text_current_slice"))
        self.button_next = QtGui.QPushButton(TestMainWindow)
        self.button_next.setGeometry(QtCore.QRect(320, 440, 151, 32))
        self.button_next.setObjectName(_fromUtf8("button_next"))
        self.text_no_slices = QtGui.QLineEdit(TestMainWindow)
        self.text_no_slices.setGeometry(QtCore.QRect(250, 440, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.text_no_slices.setFont(font)
        self.text_no_slices.setMaxLength(3)
        self.text_no_slices.setReadOnly(True)
        self.text_no_slices.setObjectName(_fromUtf8("text_no_slices"))

        self.retranslateUi(TestMainWindow)
        QtCore.QMetaObject.connectSlotsByName(TestMainWindow)

    def retranslateUi(self, TestMainWindow):
        TestMainWindow.setWindowTitle(_translate("TestMainWindow", "Form", None))
        self.button_clear_init.setText(_translate("TestMainWindow", "Clear Points", None))
        self.button_load.setText(_translate("TestMainWindow", "Load Image", None))
        self.box_scale.setText(_translate("TestMainWindow", "Scale Images", None))
        self.button_load_stk.setText(_translate("TestMainWindow", "Load Stack", None))
        self.button_previous.setText(_translate("TestMainWindow", "<- Previous Slice", None))
        self.label_5.setText(_translate("TestMainWindow", "/", None))
        self.button_go.setText(_translate("TestMainWindow", "Go!", None))
        self.button_next.setText(_translate("TestMainWindow", "Next Slice ->", None))

from MyQgraphics_widget import MyGraphicsView
