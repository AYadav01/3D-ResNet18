import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom as dicom
from os import listdir as ld
from skimage import measure, morphology, segmentation
import scipy.ndimage as ndimage


class LungSegment:
    def __init__(self, root_dir=None):
        self.root = root_dir
        if root_dir:
            self.root = root_dir
        else:
            print("Dicom Slices Required!")

    """
    Returns a 3D Array
    """
    def preprocess(self):
        if self.root:
            file_slice = []
            skipcount = 0
            for file_path in ld(self.root):
                path = os.path.join(self.root, file_path)
                ds = dicom.dcmread(path)  # will read as dicom object, not as array
                if hasattr(ds, 'SliceLocation'):
                    file_slice.append(ds)
                else:
                    skipcount += 1
            # sort the dicom objects based on slice location
            slice_arr = sorted(file_slice, key=lambda x: x.SliceLocation)
            # create a 3d array
            image = np.stack([arg.pixel_array for arg in slice_arr])
            image = image.astype(np.int16)  # convert to np.int16
            """
            set outside of scans pixels to 0
            The intercept is usually -1024, and air is apprx 0
            Anything outside of lung region is set to 0
            """
            image[image == -2000] = 0
            # Convert to HU
            intercept = slice_arr[0].RescaleIntercept
            slope = slice_arr[0].RescaleSlope
            # we make sure the slope is 1
            if slope != 1:
                image = slope * image.astype(np.float64)
                image = image.astype(np.int16)
            image += np.int16(intercept)
            return np.array(image, dtype=np.int16)
        else:
            return None

    def generate_marker(self, array):
        # creating internal marker (turn everything less than -400 "lung region is between -400 to -500")
        marker_internal = array < -400
        # remove artifacts connected to image border
        marker_internal = segmentation.clear_border(marker_internal)
        """
        Label connected regions of an integer array.
        Returns - Labeled array, where all connected regions are assigned the same integer value
        """
        marker_internal_labels = measure.label(marker_internal)
        """
        #overlay image with labels
        img_label_overlay = label2rgb(marker_internal_labels, image=array)
        plt.imshow(img_label_overlay)
        """
        """
        Measure properties of labeled image regions
        area - Returns number of pixels of the region.
        """
        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()  # sort the regions
        # since there are two lungs, we check if there are more than two regions, if yes
        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:  # if the area is smaller than the second largest values
                    # we get coordinates of that region
                    for coordinates in region.coords:
                        # we turn those cooridnate locations off
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0  # only keeps the pixels whose labels are not zero
        # generates external marker
        first_external = ndimage.binary_dilation(marker_internal, iterations=10)
        second_external = ndimage.binary_dilation(marker_internal, iterations=55)
        """
        ^ - bitwise XOR operation 
        If one or the other is a 1, it will insert a 1 in to the result, otherwise it will insert a 0. 
        This is where the name XOR, or "exclusive or" comes from.

        Digit1 | Digit2 | Result
        =========================
        0		0 		0
        0		1		1
        1		0		1
        1		1		0
        """
        marker_external = second_external ^ first_external
        # watershed matrix
        marker_watershed = np.zeros((512, 512), dtype=np.int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128
        return marker_internal, marker_external, marker_watershed

    def seperate_lungs(self, image):
        segemented_array = np.zeros(image.shape)
        marker_internal, marker_external, marker_watershed = self.generate_marker(image)
        # value of gradient (slope) in X and Y-direction
        sobel_filtered_dx = ndimage.sobel(image, 1)  # vertical derivate ( detects horizontal edges)
        sobel_filtered_dy = ndimage.sobel(image, 0)  # horizontal derivate (detects vertical edges)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)  # magnitude of gradient, gets rids of a (-)ve sign
        sobel_gradient *= 255.0 / np.max(sobel_gradient)  # normalize (This is our landscape image and we will fit it with water)
        watershed = morphology.watershed(sobel_gradient, marker_watershed)
        outline = ndimage.morphological_gradient(watershed, size=(3, 3))
        outline = outline.astype(bool)
        # Creation of the disk-kernel
        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]]
        blackhat_struct = ndimage.iterate_structure(blackhat_struct, 7)
        outline += ndimage.black_tophat(outline, structure=blackhat_struct)
        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)
        segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))
        segemented_array= segmented
        return segmented

    def plot_hist(self, array):
        image_to_show = array.astype(np.float64)
        plt.hist(image_to_show.flatten(), bins=50, color='c')
        plt.xlabel("HU")
        plt.ylabel("Frequency")
        plt.show()

    def visualize(self, array, slice_no=None):
        if slice_no is None:
            pass
        else:
            plt.imshow(array[slice_no, :, :])
        plt.show()

