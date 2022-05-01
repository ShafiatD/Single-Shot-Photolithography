# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:07:04 2022

@author: shafi
"""
import os
import cv2
from skimage import color, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from StructureAnalysisV0_3 import find_first_structure, find_structures, remove_banner_calibrate_size_bar, resize
#%%
im_path = []
# assign directory
directory = '211202 Mirrors/broken sample'

# iterate over files in
# that directory
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(file):
        im_path.append(rf'{file}')

#%%
img = cv2.imread(im_path[0], cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(image=img, threshold1=150, threshold2=400, L2gradient = True) # Canny Edge Detection

cropped_image, cropped_edges, banner, px_size, px_size_err = remove_banner_calibrate_size_bar(img, edges, show_image = True, rescale = 0.6)

array_x, array_y = 6.5e-6, 9e-6 
#%%
first_grid = find_first_structure(cropped_image,px_size,array_x,array_y,8)
#%%
structures, data = find_structures(cropped_image,cropped_edges,first_grid,array_x,array_y, px_size,
                             structures_x = 10, structures_y = 10, mode='custom')

#%%
plt.figure(dpi = 500)
full_image = np.concatenate((structures, color.gray2rgb(banner)))
plt.imshow(full_image)
plt.axis('off')
plt.tight_layout()
#%%
data = np.array(data)

# Cleaning
structure_sizes = []
for i in range(len(data)):
    structure_sizes.append(px_size*data[:,i][:,2:])
    if i == 5:
        # print(structure_sizes[i])
        structure_sizes[i] = np.delete(structure_sizes[i],0,0)
        # print(np.delete(structure_sizes[i],0,0))
        # print(structure_sizes[i])
print(structure_sizes)
#%%
# print(data)
structure_avgs = [np.average(i, axis = 0) for i in structure_sizes] # calculate average semi minor and semi major axes
structure_stds = [np.std(i, axis = 0) for i in structure_sizes] # calculate std in averaging 
structure_errs = [j/np.sqrt(len(structure_sizes[i])) for i, j in enumerate(structure_stds)] # calculate err in averaging

# print(structure_avgs,'\n')
# print(structure_errs)
defocussing = np.linspace(50e-9, -50e-9,10)/1e-9
exposure = np.linspace(50e-3,150e-3,10)/1e-3

#%% semi minor axis
a_0, b_0 = np.polyfit(defocussing, np.array(structure_avgs)[:,0]/1e-9, deg = 1,
           w = np.array(structure_errs)[:,0]/1e-9)

xarr1 = np.linspace(defocussing[0], defocussing[-1])
fit1 = a_0*xarr1 + b_0

fig, (ax1, ax2) = plt.subplots(1,2, sharey = True, figsize = (12,6))
fig.suptitle(r'Fabrication at 10$\mu$W, Exposure time: 50ms')
ax1.plot()
ax1.plot(xarr1, fit1, color = '#DC267F')
ax1.errorbar(defocussing, np.array(structure_avgs)[:,0]/1e-9,
             yerr = np.array(structure_errs)[:,0]/1e-9, fmt = 'x', color = 'black')

ax1.set_title(r'Semi minor axis')
ax1.set_xlabel('Defocussing / nm')
ax1.set_ylabel('Average structure size / nm')

# structure_areas = [np.pi*i[0]*i[1] for i in structure_avgs]
# print(structure_areas)
semi_minor_grad = a_0

# semi major axis
a_1, b_1 = np.polyfit(defocussing, np.array(structure_avgs)[:,1]/1e-9, deg = 1,
           w = np.array(structure_errs)[:,1]/1e-9)

xarr2 = np.linspace(defocussing[0], defocussing[-1])
fit2 = a_1*xarr2 + b_1

ax2.plot()
ax2.plot(xarr2, fit2, color = 'red')
ax2.errorbar(defocussing, np.array(structure_avgs)[:,1]/1e-9,
             yerr = np.array(structure_errs)[:,1]/1e-9, fmt = 'x', color = 'black')

ax2.set_title(r'Semi major axis')
ax2.set_xlabel('Defocussing / nm')
# ax2.set_ylabel('Average structure size (semi major axis)/ nm')

# structure_areas = [np.pi*i[0]*i[1] for i in structure_avgs]
# print(structure_areas)
semi_major_grad = a_1

plt.tight_layout()

#%%
print(semi_minor_grad/semi_major_grad)

#%%
a_2, b_2 = np.polyfit(defocussing, np.array(structure_avgs)[:,1]/1e-9, deg = 1,
           w = np.array(structure_errs)[:,1]/1e-9)

xarr1 = np.linspace(defocussing[0], defocussing[-1])
fit1 = a_2*xarr1 + b_2

fig, ax1 = plt.subplots(figsize = (6,6))
fig.suptitle(r'Fabrication at 10$\mu$W, Exposure time: 50ms')
ax1.plot()
ax1.plot(xarr1, fit1, color = '#DC267F')
ax1.errorbar(defocussing, np.array(structure_avgs)[:,1]/1e-9,
             yerr = np.array(structure_errs)[:,1]/1e-9, fmt = 'x', color = 'black')

ax1.set_title(r'Semi minor axis')
ax1.set_xlabel('Defocussing / nm')
ax1.set_ylabel('Average structure size / nm')

plt.tight_layout()

#%%
dim1, dim2 = img.shape
plt.figure(dpi = 200)

structure_areas = np.array([np.pi*i[0]*i[1] for i in structure_avgs])
structure_area_per_errs = np.sum(np.array(structure_errs)/np.array(structure_avgs),axis=1)

structure_diameters = np.sqrt(structure_areas / np.pi) / 1e-6
structure_dia_err = structure_diameters * structure_area_per_errs * 0.5

a_3, b_3 = np.polyfit(exposure, structure_diameters, deg = 1,
                      w = structure_dia_err)

xarr3 = np.linspace(exposure[0], exposure[-1])
fit3 = a_3*xarr3 + b_3

plt.title(r'Fabrication at 10$\mu$W', fontsize = 24)
plt.plot(xarr3, fit3, color = '#DC267F')
plt.errorbar(exposure, structure_diameters, fmt = 'x', color = 'black',
             yerr = structure_dia_err, ms = 8, mew = 1)

plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.xlabel(r'Exposure time / $ms$', fontsize = 22)
plt.ylabel(r'Average structure diameter / $\mu m$', fontsize = 22)
plt.tight_layout()
#%%
dim1, dim2 = img.shape
plt.figure(dpi = 300)

structure_areas = np.array([np.pi*i[0]*i[1] for i in structure_avgs])
structure_area_per_errs = np.sum(np.array(structure_errs)/np.array(structure_avgs),axis=1)

structure_diameters = np.sqrt(structure_areas / np.pi) / 1e-6
structure_dia_err = structure_diameters * structure_area_per_errs * 0.5

a_3, b_3 = np.polyfit(exposure, structure_diameters, deg = 1,
                      w = structure_dia_err)

xarr3 = np.linspace(exposure[0], exposure[-1])
fit3 = a_3*xarr3 + b_3

extrap_x = np.linspace(0, 50)
extrap_fit = a_3*extrap_x + b_3

plt.plot(extrap_x, extrap_fit, color = 'tab:red', linestyle = '--')
plt.plot(xarr3, fit3, color = 'tab:red', linestyle = '--')
plt.errorbar(exposure, structure_diameters, fmt = 'x', color = 'black',
             yerr = structure_dia_err, ms = 8, mew = 1)

plt.ylim(.4, 1.6)
plt.xlim(left = 0)
ylim1, ylim2 = plt.ylim()

threshreg = np.linspace(0,30)
plt.fill_between(threshreg, ylim1,ylim2, color = 'grey', alpha = 0.2, 
                 edgecolor = 'none')

# plt.title(r'Fabrication at 10$\mu$W', fontsize = 16)

plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.xlabel(r'Exposure time / $ms$', fontsize = 22)
plt.ylabel(r'Average structure diameter / $\mu m$', fontsize = 22)
plt.tight_layout()

#%%
print(4 / (a_3*40 + b_3) * (a_3*30 + b_3))
print(a_3*30 + b_3)