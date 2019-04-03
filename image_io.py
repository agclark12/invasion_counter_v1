#!/opt/local/bin/python

__author__ = "Andrew G. Clark"
__date__ = "11 February 2014"
__copyright__ = "Copyright 2014, UCL MRC-LMCB"
__maintainer__ = "Andrew G. Clark"
__email__ = "a.g.clark@ucl.ac.uk"
__status__ = "Production"

""" Methods for loading images and settings/data for invasion counter.

"""

from copy import deepcopy
import string

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.io._plugins import tifffile_plugin as tifffile

class Im:

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
        if self.scaling=='Slices':
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

def save_data_array(array, save_path):
    "Saves an even array as a justified text file"

    #Finds column widths
    column_width_list = []
    for column in zip(*array):
        column = map(str,column)
        column_width = max(len(x) for x in column) + 2
        column_width_list.append(column_width)

    #Saves array to file
    ofile = open(save_path,'w')
    for i in range(len(array)):
        for j in range(len(array[i])):
            element = string.ljust(str(array[i][j]), column_width_list[j])
            ofile.write(element + '  ')
        ofile.write('\n')
    ofile.close()

def read_file(filename, startline=0):
    """Reads an input text file into a 2D list"""

    datalist = []
    ifile = open(filename,'rU')

    for line in ifile:
        data = line.split()
        datalist.append(data)

    ifile.close()

    return datalist[startline:]

def load_spheroid_image(contour_path):

    contour_img = Im(contour_path,scaling=None)
    contour = np.array(contour_img.stk_array > 0, dtype="uint8")

    #makes mask (for segmenting nuclei)
    filled_array = np.zeros(shape=contour.shape, dtype=np.uint8)
    for i in range(contour.shape[0]):
        filled_array[i] = binary_fill_holes(contour[i])
    mask = binary_fill_holes(filled_array)

    return contour, mask

def load_nuclei_image(centroid_path):

    centroid_img = Im(centroid_path,scaling=None)
    centroid_array = np.array(centroid_img.stk_array > 0, dtype="uint8")
    pixels = np.where(centroid_array==1)
    centroids = np.zeros(shape=(len(pixels[0]),3))
    for i in range(len(pixels[0])):
        centroids[i][2] = pixels[0][i] #z (changes order)!
        centroids[i][1] = pixels[1][i] #y (changes order)!
        centroids[i][0] = pixels[2][i] #x (changes order)!

    dot_pos_list_stk = get_dot_pos_list_stk(centroids)

    return centroids, dot_pos_list_stk

def load_nuclei_list(centroid_path):

    centroid_data = read_file(centroid_path,startline=1)
    centroids = np.zeros(shape=(len(centroid_data),3))
    for i in range(len(centroid_data)):
        centroids[i] = centroid_data[i] #read in as x,y,z
    dot_pos_list_stk = get_dot_pos_list_stk(centroids)

    return centroids, dot_pos_list_stk

def get_dot_pos_list_stk(centroids):
    """Transforms an x,y,z list of centroids into a dot position list for a stack of images
    Note: be aware of the difference in indexing between the centroid array and dot list!"""

    dot_pos_list_stk = [[] for _ in range(len(centroids))]
    for coords in centroids:
        (dot_pos_list_stk[int(coords[2])]).append([int(coords[0]),int(coords[1])])

    return dot_pos_list_stk

def save_nuclei_centroids(im_stk,centroids,save_path):

    #saves nuclei centroids as an image
    img_centroids = np.zeros(shape=im_stk.shape, dtype="uint8")
    for i in range(len(centroids)):
        img_centroids[round(centroids[i][2])][round(centroids[i][1])][round(centroids[i][0])] = 1
    tifffile.imsave(save_path, np.array(img_centroids*255,dtype="uint8"))

    #saves nuclei centroid coordinates as a text file
    centroid_list = list(centroids)
    centroid_list.insert(0,["x","y","z"])
    save_data_array(centroid_list,save_path.replace(".tif",".dat"))

def test_load_im():

    im_path = "./test_stk/ctrl_1.tif"
    im = Im(im_path)
    tifffile.imsave(im_path.replace(".tif","_loaded.tif"), np.array(im.stk_array,dtype="uint8"))

if __name__ == '__main__':

    test_load_im()
