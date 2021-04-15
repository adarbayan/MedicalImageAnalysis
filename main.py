import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from PIL import Image
from tqdm.notebook import tqdm
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_closing, binary_opening
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from skimage.morphology import binary_erosion, binary_dilation, disk, square
from skimage.morphology import remove_small_objects
from skimage.morphology import area_closing
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Press the green button in the gutter to run the script.
from collections import deque

if __name__ == '__main__':
    img = cv2.imread("im1.jpg")

    """Part1"""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    figure_size = 9  # the dimension of the x and y axis of the kernel.
    kernel = np.ones((3, 3), np.uint8)
    new_image = cv2.GaussianBlur(image, (15, 19), 0)
    edges = cv2.Canny(new_image, 0, 20)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=4)
    foreground = area_closing(dilate, 1024, connectivity=1)
    plt.imshow(foreground, cmap=plt.cm.gray)
    plt.show()
    gold = np.loadtxt("im3_gold_mask.txt")
    gold1 = np.loadtxt("im3_gold_cells.txt")

    gold_int = gold.astype(int)
    gold1_int = gold1.astype(int)

    foreground[foreground == 255] = 1
    reshaped_gold = gold_int.reshape(-1)
    reshaped1_gold = gold1_int.reshape(-1)
    print(reshaped_gold)
    print(reshaped1_gold)
    foreground_final = foreground.reshape(-1)
    """print(classification_report(reshaped_gold, foreground_final))"""

    f = plt.figure()
    f1 = f.add_subplot(1, 2, 1)
    f1.set_title("Gold Standard")
    plt.imshow(gold)
    f1 = f.add_subplot(1, 2, 2)
    f1.set_title("Estimated")
    plt.imshow(foreground)
    plt.show()




    # Perform the distance transform algorithm
    substracted_foreground = cv2.subtract(foreground, edges)
    dist = cv2.distanceTransform(substracted_foreground, cv2.DIST_L2, 0)

    plt.imshow(dist, cmap=plt.cm.gray)
    plt.show()

    foreground_actual = np.loadtxt("im1_gold_mask.txt", dtype=str)
    image_max = ndi.maximum_filter(image, size=15, mode='constant')
    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(dist, min_distance=20)
    o = cv2.connectedComponents(substracted_foreground)
    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Maximum filter')

    ax[2].imshow(image, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Peak local max')
    fig.tight_layout()
    plt.show()

    plt.imshow(o[1], cmap=plt.cm.gray)
    plt.show()
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.imshow(foreground, cmap='gray')
    plt.show()
    TP_count = 0

    for i in coordinates:
        if gold[i[0], i[1]] != 0:
            TP_count += 1
    print(TP_count)
    coordinates_reshaped = coordinates.reshape(-1)


    def regionGrowing(img, x, y, thresh, count, img_new):
        stack = deque()
        stack.append([x, y])
        # print(x,y)
        while len(stack) > 0:
            # print(len(stack))
            crr = stack.pop()
            img_new[crr[0], crr[1]] = count
            neig = find_neighboors(crr)
            for i in neig:
                if i[0] < 768 and i[1] < 1024:
                    if img_new[i[0], i[1]] == 0 and dist_oklid(x, y, i[0], i[1]) < thresh:
                        if img[i[0], i[1]] < 220:
                            stack.append(i)
        return img_new


    def dist_oklid(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


    def find_neighboors(crr):
        x = crr[0]
        y = crr[1]
        return [[x - 1, y - 1], [x, y - 1], [x + 1, y - 1], [x - 1, y], [x + 1, y], [x - 1, y + 1], [x, y + 1],
                [x + 1, y + 1]]


    cv2.waitKey(0)




    foreground_c = foreground.copy()

    for i in range(len(image)):
        for j in range(len(image[0])):
            foreground_c[i, j] = 0
            if foreground[i, j] == 0:
                image[i, j] = 255
    count = 1
    for i in coordinates:
        print(i)

        foreground_c = regionGrowing(image, i[0], i[1], 20, count, foreground_c)
        count += 1

    f = plt.figure()
    f1 = f.add_subplot(1, 2, 1)

    f1.set_title("Estimated")
    plt.imshow(foreground_c)
    f1 = f.add_subplot(1, 2, 2)
    f1.set_title("Gold Standard")
    gold = np.loadtxt("im1_gold_cells.txt")
    plt.imshow(gold)
    plt.show()
    cv2.waitKey(0)













    """" img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blue_channel = img[:, :, 0]

    # global thresholding
    ret1, th1 = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    
    median = cv.medianBlur(img_gray, 5)
    ret4, th4 = cv.threshold(median, 0, 255, cv.THRESH_BINARY +cv.THRESH_OTSU )
    th5 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv.THRESH_BINARY, 11, 2)


    # plot all the images and their histograms
    images = [img_gray, 0, th1,
              img_gray, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

    blur = cv.GaussianBlur(img_gray, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    plt.imshow(th3, cmap='gray')
    plt.show()


    kernel = np.ones((3, 3), np.uint8)
    "erosion = cv.erode(th3,kernel,iterations=1)"

    dilate = cv.dilate(th3,kernel,iterations=1)
    opening = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel, iterations=3)


    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=1)

    closing_img = cv.morphologyEx(th3, cv.MORPH_CLOSE, kernel, iterations=12)
    bw_close = binary_closing(closing_img, selem=disk(6))
    im_out1 = cv.dilate(th3, kernel, iterations=6)
    im_out2 = cv.erode(im_out1, kernel, iterations=7)
    bw_fill1 = binary_fill_holes(im_out1)
    plt.imshow(bw_fill1, cmap='gray')
    plt.show()


    im_floodfill = th3.copy()
    h, w = th3.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = th3 | im_floodfill_inv
    plt.imshow(im_out, cmap='gray')
    plt.show()


    bw_fill = binary_fill_holes(im_out)
    opening = cv.morphologyEx(im_out, cv.MORPH_OPEN, kernel, iterations=1)
    im_out1 = cv.dilate(opening, kernel, iterations=6)
    im_out2 = cv.erode(im_out1, kernel, iterations=7)
    im_out3 = cv.dilate(im_out2, kernel, iterations=7)
    opening = cv.morphologyEx(im_out3, cv.MORPH_OPEN, kernel, iterations=3)
    bw_fill1 = binary_fill_holes(im_out3)
    plt.imshow(bw_fill1, cmap='gray')
    plt.show()
    cells = img[:, :, 0]
    # Threshold image to binary using OTSU. ALl thresholded pixels will be set
    # to 255

    ret1, thresh = cv.threshold(cells, 0, 255,
                                 cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Morphological operations to remove small noise - opening

    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel,
                               iterations=2)

    # finding sure background

    sure_bg = cv.dilate(opening, kernel, iterations=10)
    # applying dustance transform

    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    ret2, sure_fg =cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Now we create a marker and label the regions inside.

    ret3, markers = cv.connectedComponents(sure_fg)

    # add 10 to all labels so that sure background is not 0, but 10

    markers = markers + 10

    # Now, mark the region of unknown with zero

    markers[unknown == 255] = 0

    # applying watershed

    markers = cv.watershed(img, markers)

    # color boundaries in yellow.

    img[markers == -1] = [0, 255, 255]

    img2 = color.label2rgb(markers, bg_label=0)

    cv.imshow('Overlay on original image', img)
    cv.imshow('Colored Cells', img2)
    cv.waitKey(0)


    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)



    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    cv.imshow('image', unknown)
    cv.imshow("image1", sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    cv.imshow("image2", markers)
    cv.waitKey(0)
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    """


