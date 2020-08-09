from imageio import imread, imwrite
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from scipy.ndimage import rotate as rotate_3d
from mpl_toolkits.mplot3d import Axes3D

# Constants
GRAY = 1
RGB = 2
MAX_INTENSITY = 255
PROJECTION_FRONT = 0
PROJECTION_SIDE = 1
PROJECTION_BOTTOM = 2

"""Static functions to deal with pictures - preprocessing etc"""

# ************************************* IMPR functions ***************************************** #

class ImageProcessor:
    """
    @ path - a list of images paths
    @ representation - the pictures color representation
    """

    def __init__(self, folder_path, representation, dimensions):
        """
        Constructor of ImageProcessor.
        @param folder_path: Path of folder containing all images.
        @param representation: Image representation.
        @param dimensions: Tuple (H,W) of target dimension for all images.
        """
        self._representation = representation
        self.images = []
        self.dim = dimensions
        self.load_images(folder_path)

    def load_images(self, folder_path):
        """
        Load all images from given folder. Perform resizing and quantization.
        @param folder_path: Folder that contains images.
        @return: None.
        """
        # self.files = sorted(
        #     [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        self.files = sorted(os.listdir(folder_path))

        self.files = list(filter(os.path.exists, self.files))[:]
        print("found ", len(self.files), " files")
        for i, file in enumerate(self.files):
            image = self.read_image(file, self._representation)
            image_resized = resize(image, (self.dim[0], self.dim[1]), anti_aliasing=True)

            image_resized = rotate(image_resized, 270)
            # elif i == 1:
            #     rotated_image = rotate(image_resized, 270)
            # #elif i == 2:
            # #    rotated_image = rotate(image_resized, 90)
            image = self.quantize_image(image_resized)
            self.images.append(image)
            plt.imshow(image)
            plt.show()

    def get_image_shape(self):
        """
        @return: Return shape of first image given.
        """
        return self.images[0].shape

    def get_images(self):
        """
        @return: Return given 2d images.
        """
        return self.images

    def read_image(self, filename, representation):
        """
        Reads an image as grayscale or RGB.
        :param filename: path of image file.
        :param representation: 1 for grayscale, 2 for RGB image.
        :return: image matrix.
        """
        image = imread(filename)
        flt_image = image / MAX_INTENSITY
        if representation == GRAY:  # gray
            return rgb2gray(flt_image)
        elif representation == RGB:  # RGB
            return flt_image

    def quantize_image(self, image, threshold=0.5):
        """
        Turn image to a 2d black - nan np array.
        :param image: image file.
        :param threshold: all above will turn white, elsewhere black.
        :return: image matrix.
        """
        return np.where(image > threshold, 0, 1)

    def quantize_image_reverse(self, image, threshold=0.5):
        """
        Turn image to a 2d black - nan np array.
        :param image: image file.
        :param threshold: all above will turn white, elsewhere black.
        :return: image matrix.
        """
        return np.where(image > threshold, 1, 0)


    def get_image_projection(self, three_d_matrix, axis=0):
        """
        Project the matrix on a specific axis.
        @param three_d_matrix: 3D binary matrix.
        @param axis: axis of projection, TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK.
        @return: Projection of matrix on a given axis.
        """
        return np.any(three_d_matrix, axis).astype(int)

    def get_image_projection_complex(self, three_d_matrix, angle_x, angle_y, angle_z, axis=0,):
        """
        Project the matrix on a specific axis, after a given rotation in axis.
        @param three_d_matrix: 3D binary matrix.
        @param axis: axis of projection, TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK.
        @return: Projection of matrix on a given axis.
        """
        rotated_matrix = rotate_matrix_3d(three_d_matrix, angle_x, angle_y, angle_z)
        any = np.any(rotated_matrix>0.5, axis).astype(int)
        return any

    def crop_image(self):
        pass

    def create_image(self, shape):
        pass

    def loss_pyramid(self, three_d_matrix):
        """
        Calculate loss by l2 norm for 4 faced pyramid.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        projection_dict = {0:(0,0,0,0), 1:(0,0,120,0), 2:(0,0,-120,0), 3:(0,0,-90,PROJECTION_BOTTOM)}
        # projection_dict = {0:(0,0,0,0), 1:(0,0,-90,2)}
        for i in range(len(self.images)):
            angle_x, angle_y, angle_z, axis = projection_dict[i]
            sum += diff_punish_outliers(self.images[i], self.quantize_image_reverse(self.get_image_projection_complex(
                three_d_matrix, angle_x, angle_y, angle_z, axis)))
        # sum += 0.5*np.sum(three_d_matrix)
        return sum

    def loss_2(self, three_d_matrix):
        """
        Calculate loss by l2 norm for all 3-axis projections.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        for i in range(len(self.images)):
            sum += diff_norm_2(self.images[i], self.get_image_projection(
                three_d_matrix, i))
        return sum

    def loss_1(self, three_d_matrix):
        """
        Calculate loss by l1 norm for all 3-axis projections.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        for i in range(len(self.images)):
            sum += diff_norm_1(self.images[i], self.get_image_projection(
                three_d_matrix, i))
        return sum

    def loss_outliers_punisher(self, three_d_matrix):
        """
        Calculate loss by outliers punisher loss for all 3-axis projections.
        @param three_d_matrix: 3d binary matrix.
        @return: sum of loss along 3 different axis.
        """
        sum = 0
        for i in range(len(self.images)):
            sum += diff_punish_outliers(self.images[i], self.get_image_projection(
                three_d_matrix, i)) # + 0.5*count_empty_pixels(self.get_image_projection(three_d_matrix, i))
        return sum

    def export_result_csv(self, result, file_name):
        """
        Export result to a csv file in order to process later with rhino & GH.
        @param result: 3d binary matrix.
        @param file_name: output csv filename.
        @return: None.
        """
        data = np.where(result > 0)
        # Write the array to disk
        with open(file_name + '.csv', 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            for data_slice in data:
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_slice, delimiter=',', fmt='%-7.2f')

                # Writing out a break to indicate different z slices
                outfile.write('#\n')

def diff_norm_1(given_img, projected_img):
    """
    Calculate loss by l1 norm.
    """
    diff = given_img - projected_img
    return np.count_nonzero(diff)

def rotate_matrix_3d(three_d_matrix, angle_x, angle_y, angle_z):
    if angle_x != 0:
        three_d_matrix = rotate_3d(three_d_matrix, angle_x, axes=(1,2), reshape=False)
    if angle_y != 0:
        three_d_matrix = rotate_3d(three_d_matrix, angle_y, axes=(0, 2), reshape=False)
    if angle_z != 0:
        three_d_matrix = rotate_3d(three_d_matrix, angle_z, axes=(0, 1), reshape=False)
    return three_d_matrix

def diff_norm_2(given_img, projected_img):
    """
    Calculate loss by l2 norm.
    """
    diff = given_img - projected_img
    return np.sqrt(np.sum(abs(diff)))

def diff_punish_outliers(given_img, projected_img):
    """
    Count number of outliers, pixels that
    """
    return np.count_nonzero(given_img<projected_img)

def count_empty_pixels(projected_img):
    """
    Count number of empty pixels.
    """
    return np.count_nonzero(projected_img==0)

def print_3d_matrix(three_d_matrix, rotated_matrix):
    x, y, z = np.where(three_d_matrix != 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='blue')
    x, y, z = np.where(rotated_matrix != 0)
    ax.scatter(x, y, z, c='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 2)
    plt.ylim(0,2)
    plt.title("xy - 45 degrees")
    fig.show()

if __name__ == '__main__':
    folder_path = "images/empty"
    processor = ImageProcessor(folder_path, GRAY, folder_path)
    three_d_matrix = np.array([[[10,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,10]],[[0,0,0],[0,0,0],[0,0,0]]])
    rotated_matrix = rotate_matrix_3d(three_d_matrix,0,0,90)
    print_3d_matrix(three_d_matrix, rotated_matrix)

