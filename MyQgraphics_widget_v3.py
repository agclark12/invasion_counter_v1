#!/opt/local/bin/python

__author__ = "Andrew G. Clark"
__date__ = "2015"
__copyright__ = "Copyright 2015, Andrew Clark"
__maintainer__ = "Andrew G. Clark"
__email__ = "andrew.clark@curie.fr"
__status__ = "Production"

""" Custom widget for displaying and interacting with
images using PyQT4 for invasion segmentation

v2:
double-click to remove points
removed polygon functions

v3:
centers the image instead of just placing it or scaling it


"""

from PyQt4 import QtCore, QtGui
import numpy as np


class Dot(QtGui.QGraphicsEllipseItem):

    def __init__(self,widget,x,y,dot_size):

        self.dot_size = dot_size
        self.x_real = x
        self.y_real = y
        self.x_drawn = x - self.dot_size/2
        self.y_drawn = y - self.dot_size/2

        dot = QtGui.QGraphicsEllipseItem(self.x_drawn,self.y_drawn,dot_size,dot_size)
        self.widget = widget

        QtGui.QGraphicsEllipseItem.__init__(self)
        self.setRect(dot.rect())
        self.setBrush(QtCore.Qt.magenta)
        self.setPen(QtCore.Qt.magenta)

        self.setZValue(2)
        self.setAcceptHoverEvents(True)
        self.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

    def hoverEnterEvent(self, e):
        self.setBrush(QtCore.Qt.cyan)
        self.setPen(QtCore.Qt.cyan)

    def hoverLeaveEvent(self, e):
        self.setBrush(QtCore.Qt.magenta)
        self.setPen(QtCore.Qt.magenta)

    def mouseMoveEvent(self, e):
        new_x = e.pos().x() - self.dot_size / 2
        new_y = e.pos().y() - self.dot_size / 2

        self.setRect(new_x, new_y,
                     self.rect().width(),
                     self.rect().height())

        self.widget.update_dot_positions()

    def mouseDoubleClickEvent(self, e):
        self.widget.remove_dot(self)

    def get_real_coords(self):
        return self.rect().x() + self.dot_size / 2, self.rect().y() + self.dot_size / 2

class MyGraphicsView(QtGui.QGraphicsView):

    def __init__(self,parent=None):

        QtGui.QGraphicsView.__init__(self, parent)

        #builds scene and view
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.scene = QtGui.QGraphicsScene()
        self.setScene(self.scene)
        self.view = QtGui.QGraphicsView(self.scene)
        self.setMouseTracking(False)

        #turns off scroll bars
        self.setHorizontalScrollBarPolicy(1)
        self.setVerticalScrollBarPolicy(1)

        #adjust filters and flags
        self.scene.installEventFilter(self)
        self.setWindowFlags(self.windowFlags() |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint)

        #initializes some variables
        self.cutline = None
        self.dot_list = []
        self.img_max_length = None
        self.c_view = None
        self.dot_size = 6

        # self.setCursor(QtCore.Qt.CrossCursor)


        # This makes the view OpenGL-accelerated. Usually makes
        # things much faster, but it *is* optional.
        #self.ui.view.setViewport(QtOpenGL.QGLWidget())
        #self.setRenderHints(QtGui.QPainter.Antialiasing)

    def eventFilter(self,target,e):

        if e.type()==QtCore.QEvent.GraphicsSceneMousePress:
            pos = e.scenePos()
            item = self.itemAt(pos.x(),pos.y())

            if isinstance(item,QtGui.QGraphicsPixmapItem):
                self.add_iter_point(pos.x(),pos.y())
                return True
            else:
                return False
        else:
            return False

    def set_img_window_params(self,h,w):

        #sets frame location/size parameters
        self.img_x_min = self.x()
        self.img_y_min = self.y()
        self.w = self.width()
        self.h = self.height()

        self.img_width = w + 4 #to account for borders (border width = 2)
        self.img_height = h + 4 #to account for borders (border width = 2)
        self.img_x_max = self.img_x_min + self.img_width
        self.img_y_max = self.img_y_min + self.img_height

        #-2 accounts for width of frame around image
        if self.img_width <= self.img_height:
            self.img_max_length = self.img_width - 2
        else:
            self.img_max_length = self.img_height - 2

    def disp_img(self,gray_data,h,w,blue_data=None):
        """Displays the image on the widget.

        gray_data is a 2D numpy array with the intensity data
        w and h are integers for the image width and height, respectively

        """

        self.reset_scene()

        #fills in blue data if it's None
        if blue_data is None:
            blue_data = np.zeros((h,w),dtype=np.uint8)

        #converts to numpy arrays
        gray_data = np.array(gray_data,dtype=np.uint8)
        blue_data = np.array(blue_data,dtype=np.uint8)

        #fill in intensity values
        # (same intensity for all colors to get gray scale image)
        total = np.zeros((h,w,4),np.uint8)
        if np.max(blue_data)!=0: #add/subtract blue_data from gray_data with (0,255) saturation
            total[...,0] = (gray_data.astype(np.int16) + blue_data).clip(0, 255).astype(np.uint8) #blue
            total[...,1] = (gray_data.astype(np.int16) - blue_data).clip(0, 255).astype(np.uint8) #green
            total[...,2] = (gray_data.astype(np.int16) - blue_data).clip(0, 255).astype(np.uint8) #red
        else:
            total[...,0] = gray_data #blue
            total[...,1] = gray_data #green
            total[...,2] = gray_data #red
        total[...,3] = 255 * np.ones((h,w),dtype=np.uint8) #alpha

        #make QImage from grayscale data
        self.img = QtGui.QImage(total.data,w,h,QtGui.QImage.Format_RGB32)
        # self.img.ndarray = total #for keeping track of displayed data

        #convert to QPixmap to dispay on scene
        self.img_pixmap = QtGui.QPixmap()
        self.img_pixmap.convertFromImage(self.img)
        self.populate_img()
        #self.image_disp = self.scene.addPixmap(self.img_pixmap)

    def populate_img(self):

        # load scaled image data (xy dimensions no longer scaled - otherwise this messes with the centroid positions)
        # self.c_view = self.img_pixmap.scaled(self.img_max_length, self.img_max_length,
        # self.c_view = self.img_pixmap.scaled(self.img_width, self.img_height,
        #                                      QtCore.Qt.KeepAspectRatio,
        #                                      QtCore.Qt.FastTransformation)

        self.c_view = self.img_pixmap

        size_img = self.c_view.size()
        width, height = QtCore.QSize.width(size_img), QtCore.QSize.height(size_img)
        # print width, height

        #centers image in widget
        x_frame = self.img_x_min + self.w / 2 - self.img_width / 2 + 1 #+1 accounts for frame width
        y_frame = self.img_y_min + self.h / 2 - self.img_height / 2 + 1  #+1 accounts for frame width
        x_length = width + 2  #-2 accounts for frame width
        y_length = height + 2  #-2 accounts for frame width

        # x_frame = self.img_x_min + (self.img_x_max - self.img_x_min) / 2 - (width / 2)
        # y_frame = self.img_y_min + (self.img_y_max - self.img_y_min) / 2 - (height / 2)

        #adjust image frame to match image size
        self.setGeometry(QtCore.QRect(x_frame, y_frame, x_length, y_length))
        self.scene.setSceneRect(0,0,width,height)

        self.c_view = QtGui.QGraphicsPixmapItem(self.c_view)
        self.c_view.setZValue(0)
        self.scene.addItem(self.c_view)

        self.init_dot_positions()
        self.setFocus()

    def reset_scene(self):

        if self.dot_list:
            for dot in self.dot_list:
                self.scene.removeItem(dot)
            self.dot_list = None
            self.dot_item_list = None
        if self.c_view:
            self.scene.removeItem(self.c_view)

    def init_dot_positions(self):

        if self.dot_list:
            for dot in self.dot_list:
                self.scene.removeItem(dot)
            self.dot_list = None
            self.dot_item_list = None

        if self.dot_pos_list:
            for element in self.dot_pos_list:
                self.add_iter_point(element[0],element[1])

        self.update_dot_positions()

    def update_dot_positions(self):

        #generates list of dot positions
        self.dot_pos_list = []
        if self.dot_list:
            for element in self.dot_list:
                # x = element.rect().x() # + self.dot_size / 2 #to account for dot_size
                # y = element.rect().y() #+ self.dot_size / 2 #to account for dot_size
                x,y = element.get_real_coords()
                self.dot_pos_list.append([x,y])
                if not element in self.scene.items():
                    self.scene.addItem(element)

        #sends and updated list of the dot positions back to the parent ui
        self.parent().update_dot_positions(self.dot_pos_list)

    def redraw_dots(self,visible=True):

        for dot in self.dot_list:
            dot.setVisible(visible)

    def delete_items(self):

        if self.cutline:
            self.scene.removeItem(self.cutline)
            self.cutline=None
        if self.dot_list:
            for dot in self.dot_list:
                self.scene.removeItem(dot)
            self.dot_list = None
            self.dot_item_list = None

        self.update_dot_positions()

    def add_iter_point(self,x,y):

        # x = x - self.dot_size / 2
        # y = y - self.dot_size / 2

        dot = Dot(self,x,y,self.dot_size)
        dot.setFlags(QtGui.QGraphicsItem.ItemIsMovable)

        if self.dot_list==None:
            # self.dot_list = [dot]
            self.dot_list = [dot]
        else:
            for element in self.dot_list:
                self.scene.removeItem(element)
            self.dot_list.append(dot)

        self.update_dot_positions()

    def remove_dot(self,dot):

        if len(self.dot_list)==1:
            self.delete_items()
        else:
            self.scene.removeItem(dot)
            self.dot_list.remove(dot)
            self.update_dot_positions()

    def keyPressEvent(self, e):
        key = e.key()
        if key == QtCore.Qt.Key_Left:
            self.parent().prev_slice()
        if key == QtCore.Qt.Key_Right:
            self.parent().next_slice()
