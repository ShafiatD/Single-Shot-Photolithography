# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:13:02 2022

@author: shafi
"""
import matplotlib.pyplot as plt 
import numpy as np
from cycler import cycler
# Setting maplotlib parameters
colour = plt.cm.viridis(np.linspace(0, 1, 8))
params = {'font.size' : 12,
          'font.family' : 'serif',
          'font.serif' : 'Times New Roman',
          'mathtext.fontset' : 'stix',
          'figure.dpi' : 300,
          'axes.grid' : False,
          'xtick.minor.visible' : True,
          'xtick.top' : True,
          'xtick.direction' : 'in',
          'ytick.minor.visible' : True,
          'ytick.right' : True,
          'ytick.direction' : 'in',
          'axes.prop_cycle' : cycler('color', colour[::1]),
          }

plt.rcParams.update(params)

from tqdm import tqdm

import cv2
from skimage import color, img_as_ubyte
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

def resize(image, rescale = 1):
    """
    Resize an image given a rescaling value

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    rescale : float, optional
        Rescaling value. The default is 1.

    Returns
    -------
    rescaled_image : np.ndarray
        Rescaled image.

    """
    height, width = image.shape
    rescaled_image = cv2.resize(image, (int(width*rescale), int(height*rescale)))
    return rescaled_image

def remove_banner_calibrate_size_bar(image, edges, show_image = False, rescale = 1):
    """
    Removes the banner of the SEM image and returns the calibration of the size
    bar to pixel ratio.

    Parameters
    ----------
    image : np.ndarray
        Image to resize.
    edges : np.ndarray
        Edges detection of image, typically using Canny.
    show_image : Bool, optional
        Ouput plots of the cropped image and size bar calibration. 
        The default is False.
    rescale : float, optional
        Rescaling value for image plot. The default is 1.

    Returns
    -------
    cropped_image : np.ndarray
        Cropped image.
    cropped_edges : np.ndarray
        Cropped edges.
    banner : np.ndarray
        Banner.
    pixel_size : float
        The length of a pixel in m.
    pixel_size_err : float
        The absolute error on the pixel length.

    """
    # Run verticle edge detection
    sobely = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    
    # Calculate the average row pixel values
    avg_x_px = []
    for i in sobely:
        avg_x_px.append(np.average(i))
        
    # Banner filtering
    bottom_border_px = min(avg_x_px)        # Bottom border of the image is the minimum of the average_x_px
    bottom_border = np.where(avg_x_px == bottom_border_px)[0][0]    # Find the index of the bottom border

    top_border = bottom_border
    
    # Look for the top border, and make sure it's more top and bottom are more than 20 px apart
    tempi = 1 # Temporary index
    while bottom_border - top_border < 20:    
        tempi+=1
        top_border_px = sorted(avg_x_px)[tempi]
        top_border = np.where(avg_x_px == top_border_px)[0][0]

    top_border -= 10 # Hardcoded to make sure I don't have a bottom border of super high values
    
    # Crop out the border
    cropped_image = image[:top_border] 
    cropped_edges = edges[:top_border]
    
    if show_image: cv2.imshow('Cropped image', resize(cropped_image, rescale))
    if show_image: cv2.imshow('Cropped edges', resize(cropped_edges, rescale))
    
    height, width = edges.shape
    banner = image[top_border:bottom_border]
    size_bar = edges[top_border:bottom_border,int(width/4):int(2*width/4)] # Hard coded, find location of size bar
    
    if show_image: plt.imshow(size_bar)
    
    # Calculate average column pixel values in the image
    avg_y_px = []
    for i in size_bar.T:
        avg_y_px.append(np.average(i))
        
    # Characterise the noise as mean + 1std
    noise = np.average(avg_y_px) + np.std(avg_y_px)

    # Calculate the inner bar length
    temp1, temp2 = np.where(avg_y_px>noise)[0][1], np.where(avg_y_px>noise)[0][-2]
    bar_length_px1 = temp2-temp1
    
    if show_image:
        temparr = []
        for i in range(len(avg_y_px)):
            if (i > temp1 and i < temp2):
                temparr.append(150)
            else:
                temparr.append(0)

        plt.plot(temparr,'r')
    
    # Calculate the outer bar length
    temp1, temp2 = np.where(avg_y_px>noise)[0][0], np.where(avg_y_px>noise)[0][-1]
    bar_length_px2 = temp2-temp1

    if show_image:
        temparr = []
        for i in range(len(avg_y_px)):
            if (i > temp1 and i < temp2):
                temparr.append(150)
            else:
                temparr.append(0)

        plt.plot(temparr,'r')
        plt.imshow(size_bar)

    # Calibrate the size bar and the error
    pixel_size = 10e-6 / np.average([bar_length_px1,bar_length_px2])
    pixel_size_per_err = (np.abs(bar_length_px1 - bar_length_px2)) / np.average([bar_length_px1,bar_length_px2])
    pixel_size_err = pixel_size_per_err * pixel_size

    print(f'Pixel length\t{pixel_size:.2e}m\nError\t\t\t{pixel_size_err:.1e}m ({100*pixel_size_per_err:.2f}%)')
    
    return cropped_image, cropped_edges, banner, pixel_size, pixel_size_err

def find_ellipse(image, edges, threshold = 10, accuracy = 5, min_size = 15, 
                 max_size = 50, mode = 'custom', show_image = False, verbose = False,
                 pixel_size = None):
    """
    Algorithm that finds an ellipse in an image, using a randomised Hough
    transform as outlined in 
    https://www.researchgate.net/publication/258688218_Ellipse_detection_A_simple_and_precise_method_based_on_randomized_Hough_transform
    Ouputs the parameters of a fitted ellipse (in unit of pixel) if found.
    
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image containing object to fit.
    edges : np.ndarray
        Edge detectino of image.
    threshold : int, optional
        Threshold for the accumulator to use in the Hough transform.
        The default is 10.
    accuracy : double, optional
        Bin size on the minor axis used in the accumulator. The default is 5.
    min_size : int, optional
        Minimal major axis length. The default is 15.
    max_size : int, optional
        Maximal minor axis length. If None, the value is set to the half of the
        smaller image dimension.. The default is 50.
    mode : str, optional
        If 'custom', use user-defined fitting parameters. If 'default', no 
        parameters used. The default is 'custom'.
    show_image : bool, optional
        If True, output the fitting. The default is False.
    verbose : bool, optional
        If True, output the fitted ellipse parameters. The default is False.
    pixel_size : float, optional
        If verbose, pixel_size is needed to output calibrated size.
        The default is None.

    Returns
    -------
    yc : int
        Y centre of fitted ellipse.
    xc : int
        X centre of fitted ellipse.
    a : int
        Semi minor axis of fitted ellipse.
    b : int
        Semi major axis of fitted ellipse.
    cy : np.ndarry
        Y values of the fitted ellipse equation.
    cx : np.ndarry
        X values of the fitted ellipse equation.

    """
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    if mode.lower() == 'custom':
        result = hough_ellipse(edges, threshold, accuracy, min_size, max_size)
    elif mode.lower() == 'default':
        result = hough_ellipse(edges)
    
    # Removes any wrong fittings
    to_remove = []    
    for i, j in enumerate(result):
        # j = (list(j))
        # Wrong is defined here by having a or b == [0,1,2]
        if (0 in list(j)[3:5]) or (1 in list(j)[3:5]) or (2 in list(j)[3:5]):
            to_remove.append(i)
        # Wrong is defined here by having a/b < 1/3 or a/b > 3
        elif (list(j)[3]/list(j)[4] < 0.75) or (list(j)[3]/list(j)[4] > 1/.75):
            to_remove.append(i)
            
    result = np.delete(result, to_remove)

    # If left with no fittings then skip trying to fit it and return 0
    if result.size == 0:
        image_centre_x, image_centre_y = image.shape
        image_centre_x = int(round(image_centre_x/2))
        image_centre_y = int(round(image_centre_y/2))
        return image_centre_y, image_centre_x, 0, 0, image_centre_y, image_centre_x
    
    # Sort by accumulator
    result.sort(order='accumulator')
    
    # Estimated parameters for the ellipse
    best = list(result[-1])
    
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    if verbose:
        print(a, b)
        print(f'a = \t{a*pixel_size:.2e}nm\nb = \t{b*pixel_size:.2e}nm')
    
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    
    if show_image:
        # Draw the ellipse on the original image
        
        tempimage = color.gray2rgb(img_as_ubyte(image))
        tempimage[cy, cx] = (0, 0, 255)
        tempimage[yc, xc] = (0, 255, 0)
        # Draw the edge (white) and the resulting ellipse (red)
        tempedges = color.gray2rgb(img_as_ubyte(edges))
        tempedges[cy, cx] = (250, 0, 0)
        tempedges[yc, xc] = (0, 255, 0)

        fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                        sharex=True, sharey=True)
        
        ax1.set_title('Original picture')
        ax1.imshow(tempimage)
        
        ax2.set_title('Edge (white), result (red), centre (green)')
        ax2.imshow(tempedges)
        
        plt.show()
    
    return yc, xc, a, b, cy, cx

def find_first_structure(image, pixel_size, array_x, array_y, pixel_threshold_tol,
                         show_image = False, verbose = False):
    """
    Calculates an approxximate region where the top left structure may be.
    
    Parameters
    ----------
    image : np.ndarray
        Grayscale image containing structures.
    pixel_size : float
        If verbose, pixel_size is needed to output calibrated size.
        The default is None.
    array_x : float
        Structure spacing in x axis.
    array_y : float
        Structure spacing in y axis.
    pixel_threshold_tol : float
        Tolerance for noise cut off.
    show_image : bool, optional
        If True, outputs the found region. The default is False.
    verbose : bool, optional
        If True, outputs the centre of the region. The default is False.

    Returns
    -------
    grid_edges : list
        Edge values of the approximate region containing the first structure.

    """
    
    # Finding top left parabola

    kernal_size = 101
    gaussian_kernal = (kernal_size, kernal_size)
    image_blurred = cv2.GaussianBlur(image, gaussian_kernal, 0, 0) # Blur the image to smooth the peaks
    pixel_threshold = np.average(image_blurred)#60 # Create a threshold value

    average_y_px = [np.average(row) for row in image_blurred]
    average_x_px = [np.average(column) for column in image_blurred.T]
     
    # Plotting average pixel values with image
    fig, axs = plt.subplots(nrows=2, ncols=2)

    axs[0, 0].imshow(image)

    axs[0, 1].plot(average_y_px, list(range(len(average_y_px))))
    axs[0, 1].set_ylim(0,len(average_y_px))
    axs[0, 1].set_ylim(axs[0, 1].get_ylim()[::-1])

    axs[1, 0].plot(average_x_px)
    axs[1, 0].set_xlim(0,len(average_x_px))

    axs[1, 1].remove()

    # Setting a threshold value
    average_y_px = [0 if i < pixel_threshold - pixel_threshold_tol else i for i in average_y_px]
    average_x_px = [0 if i < pixel_threshold - pixel_threshold_tol else i for i in average_x_px]

    axs[0, 1].axvline(pixel_threshold, color = 'red', alpha = 0.5)
    axs[1, 0].axhline(pixel_threshold, color = 'red', alpha = 0.5)
    axs[0, 1].axvline(pixel_threshold - pixel_threshold_tol, color = 'red')
    axs[1, 0].axhline(pixel_threshold - pixel_threshold_tol, color = 'red')

    # Finding the centre of the first structure

    for i, pixel in enumerate(average_y_px):
        if pixel > average_y_px[i+1]:
            y_centre = i
            break

    for i, pixel in enumerate(average_x_px):
        if pixel > average_x_px[i+1]:
            x_centre = i
            break
        
    if verbose: print(f'Structure centre: \t({x_centre}, {y_centre})')

    # Draw a box around the first structure's centre given the array spacing
    # array spacing in um this needs to be changed per array so I need to save those parameters
    grid_x, grid_y = round(array_x/pixel_size), round(array_y/pixel_size)

    grid_edges = [x_centre - int(.5*grid_x), x_centre + int(.5*grid_x),
                  y_centre - int(.5*grid_y), y_centre + int(.5*grid_y)]

    for i, edge in enumerate(grid_edges):
        if edge < 0:
            grid_edges[i] = 0
    grid_edge_l, grid_edge_r, grid_edge_t, grid_edge_b = grid_edges

    if verbose: print(f'Grid edges: \t\t{grid_edges}')
    
    box = color.gray2rgb(img_as_ubyte(image))
    first_grid = np.copy(box)
    first_grid[grid_edge_t:grid_edge_b, grid_edge_l] = (255, 0, 0)
    first_grid[grid_edge_t:grid_edge_b, grid_edge_r] = (255, 0, 0)
    first_grid[grid_edge_t, grid_edge_l:grid_edge_r] = (255, 0, 0)
    first_grid[grid_edge_b, grid_edge_l:grid_edge_r] = (255, 0, 0)

    if show_image:
        plt.figure()
        plt.imshow(first_grid)
    
    return grid_edges


def find_structures(image, edges, first_grid, array_x, array_y,  pixel_size, 
                    structures_x = 10, structures_y = 10, mode = 'custom', 
                    threshold = 10, accuracy = 5, min_size = 15, max_size = 50,
                    show_image = False, verbose = False):
    """
    Function that finds all structures within an image using a sequential
    randomised elliptical Hough transform.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image containing structures.
    edges : np.ndarray
        Edge detectino of image.
    first_grid : list
        Region of the first structure in the top left of the image.
        [left edge, right edge, top edge, bottom edge].
    array_x : float
        Structure spacing in x axis.
    array_y : float
        Structure spacing in y axis.
    pixel_threshold_tol : float
        Tolerance for noise cut off.
    structures_x : int, optional
        Number of structures printed in x axis. The default is 10.
    structures_y : int, optional
        Number of structures printed in y axis. The default is 10.
    mode : str, optional
        If 'custom', use user-defined fitting parameters for Hough transform.
        If 'default', no parameters used. The default is 'custom'.
    threshold : int, optional
        Threshold for the accumulator to use in the Hough transform.
        The default is 10.
    accuracy : double, optional
        Bin size on the minor axis used in the accumulator. The default is 5.
    min_size : int, optional
        Minimal major axis length. The default is 15.
    max_size : int, optional
        Maximal minor axis length. If None, the value is set to the half of the
        smaller image dimension.. The default is 50.
    show_image : bool, optional
        If True, output show detected structures. The default is False.
    verbose : bool, optional
        If True, output fitted structure parameters. The default is False.

    Returns
    -------
    tempimage : np.ndarray
        Numpy array showing the fitted structures, the structure centres, and 
        the search regions. Pixel data is formatted as RGB.
    structure_data : np.array
        Numpy array containing the fitted ellipse parameters.
        [Ellipse centre y, Ellipse centre x, semi minor axis, semi major axis]

    """
    grid_x, grid_y = round(array_x/pixel_size), round(array_y/pixel_size)
    tempimage = color.gray2rgb(img_as_ubyte(image))
    grid_edges = first_grid
    
    structure_data = [[] for _ in range(structures_y)]
    
    with tqdm(total=structures_x*structures_y) as pbar:
        for y_arr in range(structures_y): 
            grid_edges_start = grid_edges
            grid_edge_l, grid_edge_r, grid_edge_t, grid_edge_b = grid_edges
            # print(grid_edges)
            for x_arr in range(structures_x):
                
                for i, edge in enumerate(grid_edges):
                    if edge < 0:
                        grid_edges[i] = 0
                    
                if grid_edges[3] > tempimage.shape[0]:
                    grid_edges[3] = tempimage.shape[0]
                    # print(grid_edges)
                    
                if grid_edges[1] > tempimage.shape[1]:
                    grid_edges[1] = tempimage.shape[1]
                    # print(grid_edges)
                    
                structure_img = tempimage[grid_edge_t:grid_edge_b, grid_edge_l:grid_edge_r]
                structure_edges = edges[grid_edge_t:grid_edge_b, grid_edge_l:grid_edge_r]
            
                yc, xc, a, b, cy, cx = find_ellipse(structure_img, structure_edges,
                                                    threshold = threshold, accuracy = accuracy,
                                                    min_size = min_size, max_size = max_size,
                                                    mode = mode, show_image = show_image,
                                                    verbose = verbose, pixel_size = pixel_size)
                
                ellipse_centre = [grid_edge_t + yc, grid_edge_l + xc]# = (0, 255, 0) # Plot the centre of the ellipse (Green)
                ellipse_circumference = [cy + grid_edge_t, cx + grid_edge_l]# = (0, 0, 255) # Draw ellipse around structure in full image (Blue)
            
                # Recentre box
                x_centre, y_centre = grid_edge_l + xc, grid_edge_t + yc
            
                grid_edges = [x_centre - int(.5*grid_x), x_centre + int(.5*grid_x),
                              y_centre - int(.5*grid_y), y_centre + int(.5*grid_y)]
            
                for i, edge in enumerate(grid_edges):
                    if edge < 0:
                        grid_edges[i] = 0
                    
                if grid_edges[3] > tempimage.shape[0]:
                    grid_edges[3] = tempimage.shape[0] - 1
                    print(grid_edges)
                    
                if grid_edges[1] > tempimage.shape[1]:
                    grid_edges[1] = tempimage.shape[1] - 1
                    print(grid_edges)
                    
                grid_edge_l, grid_edge_r, grid_edge_t, grid_edge_b = grid_edges
                
                if x_arr == 0:
                    grid_edges_start = grid_edges
                
                # Draw bounding box in red
                tempimage[grid_edge_t:grid_edge_b, grid_edge_l] = (255, 0, 0)
                tempimage[grid_edge_t:grid_edge_b, grid_edge_r] = (255, 0, 0)
                tempimage[grid_edge_t, grid_edge_l:grid_edge_r] = (255, 0, 0)
                tempimage[grid_edge_b, grid_edge_l:grid_edge_r] = (255, 0, 0)
            
                tempimage[tuple(ellipse_centre)] = (0, 255, 0)
                tempimage[tuple(ellipse_circumference)] = (0, 0, 255)
                
                grid_edges = [i + grid_x for i in grid_edges[:2]] + grid_edges[2:]#[i + grid_y for i in grid_edges[2:]]
                grid_edge_l, grid_edge_r, grid_edge_t, grid_edge_b = grid_edges
                
                # print('\n',grid_edges)
                # print(a, b)
                # print(f'{a*pixel_size:.2e}, {a*pixel_size:.2e}')
                
                structure_data[x_arr].append([*ellipse_centre, a, b])
                pbar.update(1)
                
            grid_edges = grid_edges_start[:2] + [i + grid_y for i in grid_edges_start[2:]]
    
    return tempimage, structure_data
            
