#!/opt/local/bin/python

__author__ = "Andrew G. Clark"
__date__ = "2015"
__copyright__ = "Copyright 2015, Andrew Clark"
__maintainer__ = "Andrew G. Clark"
__email__ = "andrew.clark@curie.fr"
__status__ = "Production"

""" Custom widget for displaying multi-layer images using PyQT4
and adding/subtracting dots

"""

from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PIL import Image
import numpy as np

class MyGraphicsView(QtGui.QGraphicsView):

    def __init__(self,parent=None):

        QtGui.QGraphicsView.__init__(self, parent)

        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.scene = QtGui.QGraphicsScene()
        self.setScene(self.scene)
        self.view = QtGui.QGraphicsView(self.scene)

        self.setMouseTracking(False)

        self.img_max_length = None
        self.setHorizontalScrollBarPolicy(1) #turn off horiz. scroll bar
        self.setVerticalScrollBarPolicy(1) #turn off vert. scroll bar

        self.scene.installEventFilter(self)

        self.cutline = None
        self.polygon = None
        self.polygon_isOpen = True

        # This makes the view OpenGL-accelerated. Usually makes
        # things much faster, but it *is* optional.
        #self.ui.view.setViewport(QtOpenGL.QGLWidget())
        #self.setRenderHints(QtGui.QPainter.Antialiasing)

    def load_img(self):

        #set frame location/size parameters (only for first image loaded)
        if self.img_max_length==None:

            self.img_x_min = self.parent().ui.widget_img.x()
            self.img_y_min = self.parent().ui.widget_img.y()
            self.img_width = self.parent().ui.widget_img.size().width()
            self.img_height = self.parent().ui.widget_img.size().height()
            self.img_x_max = self.img_x_min + self.img_width
            self.img_y_max = self.img_y_min + self.img_height

            #-2 accounts for width of frame around image
            if self.img_width <= self.img_height:
                self.img_max_length = self.img_width - 2
            else:
                self.img_max_length = self.img_height - 2

        if self.parent().ui.box_scale.checkState()==2:
            self.scaling = True
        else:
            self.scaling = False

        ipath_ = QtGui.QFileDialog.getOpenFileName(self, 'Select image stack', '.')
        ipath = str(ipath_)
        self.img_path = ipath

        # self.img_path = "/Users/aclark/Documents/Work/Programming/Python/PyScan_test/ian_crop.tif"

        self.disp_img()

    def disp_img(self):

        # self.scaling = True

        im = Image.open(self.img_path)
        # im = im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
        w = im.size[0]
        h = im.size[1]

        try:
            d = len(im.getpixel((0,0)))
        except TypeError:
            d = 1

        pil_data = im.getdata()
        data =  np.array(pil_data)
        final = np.zeros((1,h,w), dtype=np.float32) #always gray scale

        if d>1:
            for i in range(d):
                #just add colors to make grayscale if the image is color
                final[0] += np.reshape(data[...,i],(h,w))
                self.scaling=True
                # final[i] = np.reshape(data[...,i],(h,w))
        else:
            final[0] = np.reshape(np.array(data),(h,w))

        if self.scaling:
            max_int = final.max()
            min_int = final.min()
            scale_factor = 255./(max_int - min_int)
            gray_data = np.array((final[0] - min_int)*scale_factor,dtype=np.uint8)
        else:
            gray_data = np.array(final[0],dtype=np.uint8)

        #fill in intensity values
        # (same intensity for all colors to get gray scale image)
        total = np.zeros((h,w,4),np.uint8)
        total[...,0] = gray_data
        total[...,1] = gray_data
        total[...,2] = gray_data
        total[...,3] = 255 * np.ones((h,w),dtype=np.uint8)

        #make QImage from grayscale data
        self.img = QtGui.QImage(total.data,w,h,QtGui.QImage.Format_RGB32)
        self.img.ndarray = total #for keeping track of displayed data

        #convert to QPixmap to dispay on scene
        self.img_pixmap = QtGui.QPixmap()
        self.img_pixmap.convertFromImage(self.img)
        self.populate_img()
        #self.image_disp = self.scene.addPixmap(self.img_pixmap)

    def populate_img(self):

        #load scaled image data (xy dimensions scaled)
        self.c_view = self.img_pixmap.scaled(self.img_max_length, self.img_max_length,
                                             QtCore.Qt.KeepAspectRatio,
                                             QtCore.Qt.FastTransformation)
        size_img = self.c_view.size()
        width, height = QtCore.QSize.width(size_img), QtCore.QSize.height(size_img)
        x_frame = self.img_x_min + (self.img_x_max - self.img_x_min) / 2 - (width / 2)
        y_frame = self.img_y_min + (self.img_y_max - self.img_y_min) / 2 - (height / 2)

        #adjust image frame to match image size
        self.parent().ui.widget_img.setGeometry(QtCore.QRect(x_frame, y_frame, width, height))
        self.scene.setSceneRect(0,0,width-2,height-2) #-2 accounts for frame width

        self.c_view = QtGui.QGraphicsPixmapItem(self.c_view)
        self.c_view.setZValue(0)
        self.scene.addItem(self.c_view)

    def delete_items(self):
        if self.cutline:
            self.scene.removeItem(self.cutline)
            self.cutline=None
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
            self.polygon = None
        if self.dot_list:
            for dot in self.dot_list:
                self.scene.removeItem(dot)
            self.dot_list = None
            self.dot_item_list = None
        self.polygon_isOpen = True

    def finish_init(self):

        self.get_qpen()

        if self.polygon!=None:
            self.polygon_isOpen = False
            self.scene.removeItem(self.polygon_item)
            self.draw_polygon()
        else:
            pass

    def eventFilter(self,target,e):

        if e.type()==QtCore.QEvent.GraphicsSceneMousePress:
            pos = e.scenePos()
            item = self.itemAt(pos.x(),pos.y())

            if isinstance(item,QGraphicsPixmapItem):
                self.add_iter_point(pos.x(),pos.y())
                return True

            else:
                return False
        else:
            return False

    def add_iter_point(self,x,y):

            self.dot_size = 6

            x = x - self.dot_size / 2
            y = y - self.dot_size / 2

            dot = Dot(self,x,y,self.dot_size)
            dot.setFlags(QGraphicsItem.ItemIsMovable)

            if self.polygon==None:
                self.dot_list = [dot]

            else:
                for element in self.dot_list:
                    self.scene.removeItem(element)
                self.scene.removeItem(self.polygon_item)

                self.dot_list.append(dot)

            self.update_dot_positions()
            self.draw_polygon()

    def update_dot_positions(self):

        #generates list of dot positions
        self.dot_pos_list = []
        for element in self.dot_list:
            x = element.rect().x() + self.dot_size / 2
            y = element.rect().y() + self.dot_size / 2
            self.dot_pos_list.append([x,y])
            if not element in self.scene.items():
                self.scene.addItem(element)

    def draw_polygon(self):

        if self.dot_pos_list:

            self.get_qpen()
            self.polygon = QtGui.QPainterPath()

            p = QtCore.QPointF(self.dot_pos_list[0][0],
                               self.dot_pos_list[0][1])
            self.polygon.moveTo(p)

            for dot_pos in self.dot_pos_list:
                p = QtCore.QPointF(dot_pos[0],
                                   dot_pos[1])
                self.polygon.lineTo(p)

            if not self.polygon_isOpen:
                self.polygon.closeSubpath()

            width = int(self.parent().ui.box_lw.value())
            self.polygon_item = Polygon(self,self.polygon,width)

            self.scene.addItem(self.polygon_item)

    def get_qpen(self):

        self.qpen = QtGui.QPen()
        self.qpen.setColor(QColor('red'))
        self.qpen.setWidth(int(self.parent().ui.box_lw.value()))

        return self.qpen