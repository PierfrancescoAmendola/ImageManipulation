import pandas as pd 
import numpy as np

from glob import glob

import cv2
import matplotlib.pylab as plt

#---------Reading image---------

appleFiles = glob('../images/apple fruit/*.jpg')
bananaFiles = glob('../images/banana fruit/*.jpg')

#f---------unziona di matploit per leggere---------
img_mpl = plt.imread(appleFiles[20])

#---------funzione di cv2 per leggere---------
img_cv2 = cv2.imread(appleFiles[20])

#---------serve per stampare la dimensione dell'immagine con mat---------
print(img_mpl.shape) #uguali
print(img_cv2.shape)

#---------image array (height, width, channels)---------
print(img_mpl)

pd.Series(img_mpl.flatten()).plot(kind= 'hist', bins = 50, title = 'Distribution of Pixel Values')
#mostra il grafico fi distribuzione dei pixel
plt.show()

#---------Display images---------
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_mpl)
plt.show()


#---------Image Channels---------
#Display RGB channels of our image
#Tre rappresentazioni della stessa immagine
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_mpl[:,:,0], cmap='Reds')
axs[1].imshow(img_mpl[:,:,1], cmap='Greens')
axs[2].imshow(img_mpl[:,:,2], cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[0].set_title('Red Channel')
axs[1].set_title('Green Channel')
axs[2].set_title('Blue Channel')
plt.show()

#Maptloit read in RGB, instead cv2 read in BGR
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_cv2)
axs[1].imshow(img_mpl)
axs[0].set_title('CV image')
axs[1].set_title('Matploitlib image')
plt.show()

#convertiamo ora da BGR a  RGB
img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots()
ax.imshow(img_cv2_rgb)
ax.set_title('CV image converted')
plt.show()

#---------Image manipulation---------

img = plt.imread(bananaFiles[4])
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.axis('off')
plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_gray, cmap='Greys')
ax.axis('off')
ax.set_title('Grey image')
plt.show()

#---------Resizing and scaling---------
img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_resized)
ax.axis('off')
plt.show()

#altro resize ma questo non sarà pixelato

img_2_resized = cv2.resize(img, (100, 200))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_2_resized)
ax.axis('off')
plt.show()


image_resized_scalde = cv2.resize(img, (5000, 5000), interpolation = cv2.INTER_CUBIC)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_resized_scalde)
ax.axis('off')
plt.show()

#Blur an image

kernel_3x3 = np.ones((3, 3), np.float32) / 9
blurred = cv2.filter2D(img_mpl, -1, kernel_3x3)
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(blurred)
ax.axis('off')
ax.set_title('Blurred image')
plt.show()


#---------Save image---------

plt.imsave('mpl_apple.png', blurred)
cv2.imwrite('cv2_apple.png', blurred)


# comando python /Users/pierfrancesco/Desktop/ImagePython/prova.py
