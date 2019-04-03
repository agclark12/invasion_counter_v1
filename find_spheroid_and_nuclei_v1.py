#!/opt/local/bin/python

__author__ = "Andrew G. Clark"
__date__ = "2015"
__copyright__ = "Copyright 2015, Andrew Clark"
__maintainer__ = "Andrew G. Clark"
__email__ = "andrew.clark@curie.fr"
__status__ = "Production"

"""

Methods for finding spheroid border and nuclei and plotting.

"""

from copy import deepcopy

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pylab
from scipy.ndimage import label,gaussian_filter
from scipy.ndimage.morphology import binary_erosion,binary_dilation, binary_fill_holes
from scipy.ndimage.measurements import sum as ndimagesum
from skimage.filters import threshold_otsu
from skimage.io._plugins import tifffile_plugin as tifffile
from skimage.morphology import remove_small_objects
import skimage._shared.geometry #hidden (prevents packaging import error)
import skimage.filters.rank.core_cy #hidden (prevents packaging import error)
import skimage.io

from image_io import Im, save_data_array

def get_nuclei(img_array,mask,im_path,stk_hist=False,sigma=2,thresh_type='Moments',
               thresh_factor=1.5,min_size=250,output_toggle=False):

    se = get_spherical_se(radius=1)
    thresh_factor = float(thresh_factor)

    #blur and threshold
    if stk_hist:

        #blurs image
        img_blurred = np.zeros(shape=img_array.shape, dtype=np.uint8)
        for i in range(img_blurred.shape[0]):
            img_blurred[i] = gaussian_filter(img_array[i],sigma)
        #performs threshold
        if thresh_type=='Moments':
            thresh = get_moments_thresh(img_blurred)
        elif thresh_type=='Otsu':
            thresh = threshold_otsu(img_blurred)
        else:
            raise TypeError("Invalid <<thresh_type>>! Please select <<Moments>> or <<Otsu>>")
        img_binary = np.array(img_blurred > thresh*thresh_factor, dtype="uint8")

    else:

        img_binary = np.zeros(shape=img_array.shape, dtype=np.uint8)
        for i in range(img_binary.shape[0]):
            slice_blurred = gaussian_filter(img_array[i],sigma)
            if thresh_type=='Moments':
                thresh = get_moments_thresh(slice_blurred)
            elif thresh_type=='Otsu':
                thresh = threshold_otsu(slice_blurred)
            else:
                raise TypeError("Invalid <<thresh_type>>! Please select <<Moments>> or <<Otsu>>")
            img_binary[i] = np.array(slice_blurred > thresh*thresh_factor, dtype="uint8")

    img_masked = np.array((img_binary - mask)==1, dtype="uint8")

    #gets rid of small objects
    morph_no = 2
    img_masked = binary_erosion(img_masked,structure=se,iterations=morph_no)
    img_masked = binary_dilation(img_masked,structure=se,iterations=morph_no)

    ####NOTE: I TRIED A WATERSHED ANALYSIS HERE TO SEPARATE THE NUCLEI,
    ####BUT JUST USING THRESHOLDING AND MORPHOLOGY GAVE BETTER RESULTS

    #removes objects below threshold size
    labels_cells, no_labels_cells =label(img_masked)
    labels_updated = remove_small_objects(labels_cells,min_size=250)
    img_masked = np.array(labels_updated > 0, dtype="uint8")

    if output_toggle:
        tifffile.imsave(im_path.replace(".tif","_cells_masked.tif"), np.array(img_masked*255,dtype="uint8"))

    return img_masked

def get_spheroid_contour(img_array,im_path,stk_hist=False,sigma=2,thresh_type='Moments',
                         thresh_factor=1.5,erosion_factor=5,smoothing_factor=40,output_toggle=False):

    se = get_spherical_se(radius=1)
    thresh_factor = float(thresh_factor)

    #blurs and thresholds
    if stk_hist:

        img_blurred = np.zeros(shape=img_array.shape, dtype=np.uint8)
        #blurs image
        for i in range(img_blurred.shape[0]):
            img_blurred[i] = gaussian_filter(img_array[i],sigma)
        #performs threshold
        if thresh_type=='Moments':
            thresh = get_moments_thresh(img_blurred)
        elif thresh_type=='Otsu':
            thresh = filter.threshold_otsu(img_blurred)
        else:
            raise TypeError("Invalid <<thresh_type>>! Please select <<Moments>> or <<Otsu>>")
        img_binary = np.array(img_blurred > thresh*thresh_factor, dtype="uint8")

    else:

        img_binary = np.zeros(shape=img_array.shape, dtype=np.uint8)
        for i in range(img_binary.shape[0]):
            slice_blurred = gaussian_filter(img_array[i],sigma)
            if thresh_type=='Moments':
                thresh = get_moments_thresh(slice_blurred)
            elif thresh_type=='Otsu':
                thresh = filter.threshold_otsu(slice_blurred)
            else:
                raise TypeError("Invalid <<thresh_type>>! Please select <<Moments>> or <<Otsu>>")
            img_binary[i] = np.array(slice_blurred > thresh*thresh_factor, dtype="uint8")

    if output_toggle:
        tifffile.imsave(im_path.replace(".tif","_sph_bin.tif"), np.array(img_binary*255,dtype="uint8"))

    #pads borders (for smoothing)
    pad_size = smoothing_factor * 2
    img_binary = np.pad(img_binary,pad_size,mode='reflect',reflect_type='even')

    #helps separate the spheroid body from nuclei that are close by
    img_binary = binary_erosion(img_binary,structure=se, iterations=erosion_factor)

    if output_toggle:
        tifffile.imsave(im_path.replace(".tif","_sph_bin_morph.tif"), np.array(img_binary*255,dtype="uint8"))

    #isolate biggest object
    labels_sph, no_labels_sph = label(img_binary)
    sizes = ndimagesum(img_binary,labels_sph,range(1,no_labels_sph+1))
    max_label = np.where(sizes==sizes.max())[0] + 1
    max_array = np.array(labels_sph==max_label[0], dtype="uint8")

    if output_toggle:
        tifffile.imsave(im_path.replace(".tif","_sph_isolated.tif"), np.array(max_array*255,dtype="uint8"))

    #more morphology to smooth the edges
    max_array = binary_dilation(max_array,structure=se, iterations=smoothing_factor)
    max_array = binary_erosion(max_array, structure=se, iterations=smoothing_factor)
    # max_array = ndimage.morphology.binary_erosion(max_array, structure=se, iterations=smoothing_factor+erosion_factor)

    #fills in remaining holes in contour (in 3D)
    filled_array = binary_fill_holes(max_array)

    #trim the pads off (this array will be used as a mask for the invading cells)
    filled_array_trimmed = filled_array[pad_size:-pad_size,pad_size:-pad_size,pad_size:-pad_size]

    #fills in the last frame (to avoid holes in the center) and refills volume
    filled_array_trimmed[-1] = binary_fill_holes(filled_array_trimmed[-1])
    filled_array_trimmed = binary_fill_holes(filled_array_trimmed)

    #fill in remaining holes in contour
    #(in 2D for each slice; leave this commented out for now)
    # filled_array = np.zeros(shape=max_array.shape)
    # for i in range(len(max_array)):
    #     filled_array[i] = ndimage.morphology.binary_fill_holes(max_array[i])

    #get the contours
    contours = (binary_dilation(filled_array_trimmed,structure=se) -
                binary_erosion(filled_array_trimmed, structure=se))

    #isolate biggest contour (to be sure there's only one in the end)
    labels_sph, no_labels_sph = label(contours)
    sizes = ndimagesum(contours,labels_sph,range(1,no_labels_sph+1))
    max_label = np.where(sizes==sizes.max())[0] + 1
    max_array = np.array(labels_sph==max_label[0], dtype="uint8") #the contour image

    #clears out the last frame from the contour
    contour_trimmed = deepcopy(max_array)
    se2d = [[0,1,0],[1,1,1],[0,1,0]]
    contour_trimmed[-1] = (binary_dilation(contour_trimmed[-1],structure=se2d) -
                           binary_erosion(contour_trimmed[-1], structure=se2d))

    #write contour to image
    if output_toggle:
        tifffile.imsave(im_path.replace(".tif","_sph_contour.tif"), 255*np.array(contour_trimmed,dtype="uint8"))

    return contour_trimmed, filled_array_trimmed

def get_moments_thresh(img_array):
    """http://fiji.sc/Auto_Threshold#Percentile
    for 3D image stacks"""

    #creates histogram of img_array
    hist, bins = np.histogram(img_array.ravel(),256,[0,256])
    hist = np.array(map(float,hist))

    #normalizes histogram
    total = 0.
    for i in range(256):
        total += hist[i]
    histo = hist / total #normalized histogram

    # gets first,second,third order moments
    m0, m1, m2, m3 = 1., 0., 0., 0.
    for i in range(256):
        m1 += float(i) * histo[i]
        m2 += float(i) * float(i) * histo[i]
        m3 += float(i) * float(i) * float(i) * histo[i]
    cd = m0 * m2 - m1 * m1
    c0 = ( -m2 * m2 + m1 * m3 ) / cd
    c1 = ( m0 * -m3 + m2 * m1 ) / cd
    z0 = 0.5 * ( -c1 - np.sqrt( c1 * c1 - 4.0 * c0 ) )
    z1 = 0.5 * ( -c1 + np.sqrt( c1 * c1 - 4.0 * c0 ) )
    p0 = ( z1 - m1 ) / ( z1 - z0 )  # Fraction of the object pixels in the target binary image

    # gets threshold value (gray-level closest to the p0-tile of the normalized histogram)
    total2 = 0.
    threshold = -1
    for i in range(256):
        total2+=histo[i]
        if total2 > p0:
            threshold = i
            break

    return threshold

def get_spherical_se(radius=2):

    N = (radius * 2) + 1
    se = np.zeros(shape=[N,N,N])
    for i in range(N):
        for j in range(N):
            for k in range(N):
                se[i,j,k] = (i - N / 2)**2 + (j - N / 2)**2 + (k - N / 2)**2 <= radius**2
    se = np.array(se)
    return se

def get_centroids(img,img_bin_nuclei,im_path,output_toggle=False):
    """finds centroid positions (rounded to nearest pixel) using weighted mean of intensities
    for each nucleus """

    #relabels masked image
    img_masked = np.array(img_bin_nuclei > 0, dtype="uint8")
    labels_cells, no_labels_cells = label(img_masked)

    #determine centroid positions and the pixels on the contour that are closest to the centroid
    centroids = np.zeros(shape=(no_labels_cells,3))

    #gets weighted centers of mass as centroid positions (this is still slower than it should be!)
    for i in range(1,no_labels_cells+1):

        pixels = np.where(labels_cells==i)
        intensities = img[pixels]

        # centroids[i-1][2] = np.sum((pixels[0]*intensities)) / np.sum(intensities) #z (changes order)!
        # centroids[i-1][1] = np.sum((pixels[1]*intensities)) / np.sum(intensities) #y (changes order)!
        # centroids[i-1][0] = np.sum((pixels[2]*intensities)) / np.sum(intensities) #x (changes order)!

        #gets centroids (rounded to nearest pixel!)
        centroids[i-1][2] = int(round(np.sum(pixels[0]*intensities) / np.sum(intensities))) #z (changes order)!
        centroids[i-1][1] = int(round(np.sum(pixels[1]*intensities) / np.sum(intensities))) #y (changes order)!
        centroids[i-1][0] = int(round(np.sum(pixels[2]*intensities) / np.sum(intensities))) #x (changes order)!

    if output_toggle:
        img_centroids = np.zeros(shape=img.shape,dtype="uint8")
        for i in range(len(centroids)):
            img_centroids[round(centroids[i][2])][round(centroids[i][1])][round(centroids[i][0])] = 1
        tifffile.imsave(im_path.replace(".tif","_centroids.tif"), np.array(img_centroids*255,dtype="uint8"))

    return centroids

def get_closest_contour_px(img_contour,centroids,px_size_xy=1.,px_size_z=1.):

    no_centroids = len(centroids)

    closest_contour_pixels = np.zeros(shape=(no_centroids,3))
    contour_pixels = np.where(img_contour==1)

    contour_z = np.array(contour_pixels[0])
    contour_y = np.array(contour_pixels[1])
    contour_x = np.array(contour_pixels[2])

    #scales contour pixels
    contour_x_scaled = contour_x * px_size_xy
    contour_y_scaled = contour_y * px_size_xy
    contour_z_scaled = contour_z * px_size_z


    #for each centroid position, find the closest position on the spheroid contour
    for i in range(len(centroids)):

        centroid_x = centroids[i][0]*px_size_xy
        centroid_y = centroids[i][1]*px_size_xy
        centroid_z = centroids[i][2]*px_size_z

        distances_tmp = np.sqrt((contour_z_scaled-centroid_z)**2 +
                                (contour_y_scaled-centroid_y)**2 +
                                (contour_x_scaled-centroid_x)**2)

        min_idx = np.argmin(distances_tmp)
        min_distance = distances_tmp[min_idx]

        closest_px_x = contour_x[min_idx]
        closest_px_y = contour_y[min_idx]
        closest_px_z = contour_z[min_idx]

        closest_contour_pixels[i][0] = closest_px_x
        closest_contour_pixels[i][1] = closest_px_y
        closest_contour_pixels[i][2] = closest_px_z

        # print [centroid_x,centroid_y,centroid_z]
        # print [closest_px_x,closest_px_y,closest_px_z]


    return closest_contour_pixels

def plot_invasion(img_contour,centroids,closest_contour_pixels,save_path,im_name,px_size_xy=1.,px_size_z=1.):
    """plots the contour and invading cells in 3D and writes positions to text file.
    This function also automatically saves some stats and information about the invading cells."""

    contour_pixels = np.where(img_contour==1)

    contour_z = np.array(contour_pixels[0])
    contour_y = np.array(contour_pixels[1])
    contour_x = np.array(contour_pixels[2])

    #reverses the contour pixels and shifts up (to have the top of the spheroid pointing "up")
    contour_z = -contour_z + img_contour.shape[0]

    #scales contour pixels
    contour_x = contour_x * px_size_xy
    contour_y = contour_y * px_size_xy
    contour_z = contour_z * px_size_z

    ####fits the spheroid body (the inner rim of the contour)

    #prepares the plot
    fig = pylab.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    ax.scatter(contour_x, contour_y, contour_z, zdir='z',s=0.4,facecolors='b',edgecolors='b',alpha=0.02)
    ax.view_init(elev=10., azim=37.5)

    centroids_x, centroids_y, centroids_z = zip(*centroids)
    closest_pxs_x, closest_pxs_y, closest_pxs_z = zip(*closest_contour_pixels)

    centroids_z = -np.array(centroids_z) + img_contour.shape[0] #reverses and shifts up
    closest_pxs_z = -np.array(closest_pxs_z) + img_contour.shape[0] #reverses and shifts up

    #scales centroid pixels
    centroids_x = np.array(centroids_x) * px_size_xy
    centroids_y = np.array(centroids_y) * px_size_xy
    centroids_z = centroids_z * px_size_z

    #scales closest pixels
    closest_pxs_x = np.array(closest_pxs_x) * px_size_xy
    closest_pxs_y = np.array(closest_pxs_y) * px_size_xy
    closest_pxs_z = np.array(closest_pxs_z) * px_size_z

    for i in range(len(centroids_x)):

        #plots lines going from centroid to closest position on spheroid contour
        x_line = (centroids_x[i],closest_pxs_x[i])
        y_line = (centroids_y[i],closest_pxs_y[i])
        z_line = (centroids_z[i],closest_pxs_z[i])
        ax.plot(x_line,y_line,z_line, zdir='z',c='k')

        #plots the centroid position
        ax.scatter(centroids_x[i], centroids_y[i], centroids_z[i], zdir='z', c= 'red',s=20, depthshade=False)

    #increases axis limits
    xy_lim = max([img_contour.shape[1],img_contour.shape[2]])
    ax.set_xlim(0,xy_lim*px_size_xy)
    ax.set_ylim(0,xy_lim*px_size_xy)
    ax.set_zlim(0,img_contour.shape[0]*px_size_z*1.2)

    #impose equal aspect ratio (i.e. set limits equal)
    # scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    # # ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
    # max_lims = [np.min(scaling), np.max(scaling)]
    # ax.set_xlim(max_lims[0],max_lims[1])
    # ax.set_ylim(max_lims[0],max_lims[1])
    # ax.set_zlim(max_lims[0],max_lims[1])

    # ax.set_xlim(0, ax.get_xlim()[1] + ax.get_xlim()[1] * 0.2)
    # ax.set_ylim(0, ax.get_ylim()[1] + ax.get_ylim()[1] * 0.2)
    # ax.set_zlim(0, ax.get_zlim()[1] + ax.get_zlim()[1] * 2)
    # ax.set_xlim(0,500.0000001)
    # ax.set_ylim(0,500.0000001)
    # ax.set_zlim(0,400.0000001)

    #finishes and saves the plot
    ax.set_xlabel("x-axis ($\mu$m)")
    ax.set_ylabel("y-axis ($\mu$m)")
    ax.set_zlabel("z-axis ($\mu$m)")
    ax.dist = 11 #to fit all the labels on the image
    ax.yaxis._axinfo['label']['space_factor'] = 2.3
    ax.xaxis._axinfo['label']['space_factor'] = 2.3
    pylab.savefig(save_path)
    pylab.close()

    #prepares a list for writing out statistics
    stats_to_write = [['im_name','no_inv_cells','mean_inv_dist','median_inv_dist',
                      'inv_dist_peak','no_inv_cells_norm']]

    #calculate a list of distances (this is now in um since the positions have already been converted)
    distances = np.array(np.sqrt((closest_pxs_z-centroids_z)**2 +
                                 (closest_pxs_y-centroids_y)**2 +
                                 (closest_pxs_x-centroids_x)**2))

    ####write out all of the centroid positions
    centroid_nos = np.array(range(len(centroids_x)))
    data_to_write = [centroid_nos,
                     centroids_x,centroids_y,centroids_z,
                     closest_pxs_x,closest_pxs_y,closest_pxs_z,
                     distances]
    data_to_write = zip(*data_to_write) #transpose
    data_to_write.insert(0,['centroid_no',
                            'centroid_x_um','centroid_y_um','centroids_z_um',
                            'closest_px_x_um','closest_px_y_um','closest_px_z_um',
                            'dist_um'])
    save_data_array(data_to_write,save_path.replace(".png", "_centroid_data.dat"))

    #calcualte stats
    no_inv_cells = len(distances)
    mean_inv_dist = np.mean(distances)
    median_inv_dist = np.median(distances)

    dist_hist, bin_edges = np.histogram(distances,bins=10,range=(0,distances.max()))
    inv_dist_peak = np.mean([bin_edges[np.argmax(dist_hist)],bin_edges[np.argmax(dist_hist)+1]])

    sph_sa = len(contour_x) * px_size_xy**2 #x is just the x-coordinates of the spheroid contour pixels from above
    no_inv_cells_norm = no_inv_cells / sph_sa

    stats_to_write.append([im_name,no_inv_cells,mean_inv_dist,median_inv_dist,
                          inv_dist_peak,no_inv_cells_norm])

    save_data_array(stats_to_write,save_path.replace(".png", "_stats.dat"))

    #plot a histogram of the invasion distances
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ax.hist(distances,bins=10,range=(0,distances.max()))
    ax.set_xlabel(r'Invasion Distance ($\mu$m)')
    ax.set_ylabel('Number of cells')
    ax.xaxis.labelpad = 20
    pylab.savefig(save_path.replace(".png", "_hist.png"))

def test_methods():

    #sets image params
    im_path = "./test_stk/ctrl_1.tif"
    px_size_xy = 0.8258 #in microns
    px_size_z = 1. #in microns
    output_toggle = True

    #sets spheroid parameters
    stk_hist_sph = True
    sigma_sph = 2
    thresh_type_sph = 'Moments'
    thresh_factor_sph = 1.
    erosion_factor_sph = 5
    smoothing_factor_sph = 40

    #sets nuclei parameters
    stk_hist_nuclei = True
    sigma_nuclei = 2
    thresh_type_nuclei = 'Moments'
    thresh_factor_nuclei = 1.5
    min_size_nuclei = 250

    #opens image, finds spheroid/nuclei and plots
    print "Opening Image!"
    im = Im(im_path)
    print "Finding Spheroid Contour!"
    contour,mask = get_spheroid_contour(im.stk_array, im_path,
                                        stk_hist=stk_hist_sph, sigma=sigma_sph, thresh_type=thresh_type_sph,
                                        thresh_factor=thresh_factor_sph, erosion_factor=erosion_factor_sph,
                                        smoothing_factor=smoothing_factor_sph, output_toggle=output_toggle)
    print "Finding Nuclei!"
    nuclei = get_nuclei(im.stk_array, mask, im_path,
                        stk_hist=stk_hist_nuclei, sigma=sigma_nuclei, thresh_type=thresh_type_nuclei,
                        thresh_factor=thresh_factor_nuclei, min_size=min_size_nuclei, output_toggle=output_toggle)
    print "Determining Centroid Positions!"
    centroids = get_centroids(im.stk_array,nuclei,im_path,output_toggle=output_toggle)
    print "Determining Closest Contour Pixels!"
    closest_contour_pixels = get_closest_contour_px(contour,centroids)
    print "Plotting and Saving!"
    plot_invasion(contour,centroids,closest_contour_pixels,im_path,px_size_xy,px_size_z)

if __name__ == '__main__':

    test_methods()