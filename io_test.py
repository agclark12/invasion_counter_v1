#!/opt/local/bin/python

__author__ = "Andrew G. Clark"
__date__ = ""
__copyright__ = "Copyright 2015, Andrew G. Clark"
__maintainer__ = "Andrew G. Clark"
__email__ = "andrew.clark@curie.fr"
__status__ = "Production"

"""

Test new methods for loading and saving tiff stacks

"""

from time import strftime
from copy import deepcopy

import numpy as np
from skimage.io._plugins import freeimage_plugin as fi
from PIL import Image

from skimage.io._plugins import tifffile_plugin as tifffile


class Im:

    def __init__(self,img_path,scaling=False):

        self.img_path = img_path
        self.scaling = scaling
        self.load_stk()

    def load_stk(self):

        self.im = Image.open(self.img_path)

        #checks image and gets some basic parameters
        self.check_bitdepth(self.im)
        self.no_slices = self.count_slices(self.im)
        self.im_size = (self.no_slices,self.im.size[1],self.im.size[0]) #x and y need to be swapped
        self.stk_array = np.zeros(shape=self.im_size)


        for i in range(self.no_slices):

            #adds image to self.stk_array
            self.im.seek(i)

            #opens and displays image
            w = self.im_size[2]
            h = self.im_size[1]

            try:
                d = len(self.im.getpixel((0,0)))
            except TypeError:
                d = 1

            pil_data = self.im.getdata()
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

            #performs scaling slice-by-slice
            if self.scaling=='Slice':
                max_int = final.max()
                min_int = final.min()
                scale_factor = 255./(max_int - min_int)
                gray_data = np.array((final[0] - min_int)*scale_factor,dtype=np.uint8)
            else:
                gray_data = np.array(final[0],dtype=np.uint8)

            #self.img_array contains all of the image intensity data
            self.stk_array[i] = np.array(gray_data)

        #performs scaling by stack intensity
        if self.scaling=='Stack':
            max_int = self.stk_array.max()
            min_int = self.stk_srray.min()
            scale_factor = 255./(max_int - min_int)
            self.stk_array = np.array((self.stk_array[0] - min_int)*scale_factor,dtype=np.uint8)

    def check_bitdepth(self,im):
        """Checks bitdepth of image and raises error if image is not 8- or 16-bit
        Args: im: PIL image type, any image. Returns, none.

        """

        # print im.mode
        if im.mode=="L" or im.mode=="I;16B":
            pass
        else:
            raise TypeError("Images to load must be 8- or 16-bit monochrome!" +
                            " File: %s was not imported correctly" %im.filename)

    def count_slices(self,im):
        """Counts number of slices in image stack. Args: im: PIL image
        type, image of any depth. Returns: none.

        """

        im.seek(0)
        try:
            while 1:
                slice_counter = im.tell()
                im.seek(im.tell()+1)
        except EOFError:
            pass # end of sequence

        no_slices = slice_counter + 1

        return no_slices

class Im2:

    def __init__(self,img_path,scaling=False):

        self.img_path = img_path
        self.scaling = scaling
        self.load_stk()

    def load_stk(self):

        self.stk_array = tifffile.imread(self.img_path) #read in with z,y,x coordinates

        #ensures the stk_array always have 3 dimensions, converts to grayscale by summing colors
        if self.stk_array.ndim==4: #RGB stack
            self.stk_array = np.sum(self.stk_array,axis=3)
        elif self.stk_array.ndim==3: #BW stack or RGB image
            if self.stk_array.shape[-1]==3: #RGB image (assumes you would never have width=3px)
                self.stk_array = np.sum(self.stk_array,axis=2)
                self.stk_array = np.reshape(self.stk_array,(1,self.stk_array.shape[0],self.stk_array.shape[1]))
            else: #BW stack
                self.stk_array = np.reshape(self.stk_array,self.stk_array.shape)
        elif self.stk_array.ndim==2: #BW image
            self.stk_array = np.reshape(self.stk_array,(1,self.stk_array.shape[0],self.stk_array.shape[1]))

        self.no_slices = self.stk_array.shape[0]
        self.im_size = deepcopy(self.stk_array.shape)

        #performs scaling slice-by-slice
        if self.scaling=='Slice':
            for i in range(self.no_slices):
                max_int = self.stk_array[i].max()
                min_int = self.stk_array[i].min()
                scale_factor = 255./(max_int - min_int)
                self.stk_array[i] = np.array((self.stk_array[i] - min_int)*scale_factor,dtype=np.uint8)

        #performs scaling by stack intensity
        if self.scaling=='Stack':
            max_int = self.stk_array.max()
            min_int = self.stk_array.min()
            scale_factor = 255./(max_int - min_int)
            self.stk_array = np.array((self.stk_array - min_int)*scale_factor,dtype=np.uint8)

def test_load_im():

    # im_path = "./ian_stk.tif"
    im_path = "./ctrl_1_test/ctrl_1.tif"
    # im_path = "./ctrl_1_test/ctrl_1_slice.tif"
    # im_path = "./ctrl_1_test/ctrl_1_rgb.tif"
    # im_path = "./ctrl_1_test/ctrl_1_rgb_slice.tif"

    # print(strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    # im = Im(im_path)
    # # fi.write_multipage(np.array(im.stk_array,dtype="uint8"), im_path.replace(".tif","_loaded.tif"))
    # print(strftime("%Y-%m-%d_%Hh%Mm%Ss"))

    print(strftime("%Y-%m-%d_%Hh%Mm%Ss"))
    im2 = Im2(im_path)
    tifffile.imsave(im_path.replace(".tif","_loaded_tifffile.tif"), np.array(im2.stk_array,dtype="uint8"))
    print(strftime("%Y-%m-%d_%Hh%Mm%Ss"))

if __name__ == '__main__':

    test_load_im()

