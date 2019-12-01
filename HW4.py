import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

NUM_ROW = 6
NUM_COL = 6
NUM_BLOCK = 36


def suffleImage(image, block_size_x, block_size_y):
    image2 = np.array(image)
    image2 = image2[0:NUM_ROW*block_size_y, 0:NUM_COL*block_size_x, 0:3]

    block_images = np.zeros((block_size_y, block_size_x, 3, NUM_BLOCK))
    suffle_image = np.zeros(
        (NUM_ROW*block_size_y, NUM_COL*block_size_x, 3), dtype='uint8')
    index = np.random.permutation(NUM_BLOCK)

    for i in range(0, NUM_ROW):
    	for j in range(0, NUM_COL):
    		ymin = i * block_size_y
    		xmin = j * block_size_x
    		indextmp = index[i * NUM_ROW + j]
    		block_images[:, :, :, indextmp] = image2[ymin:ymin +
                                               block_size_y, xmin:xmin+block_size_x, :]
    		y = math.floor(indextmp/NUM_ROW)
    		x = indextmp - y * NUM_ROW
    		suffle_image[y*block_size_y:(y+1)*block_size_y, x*block_size_x:(
    		    x+1)*block_size_x, ::-1] = block_images[:, :, :, indextmp]

    return [suffle_image, block_images]


image = Image.open('./cat.jpg')
[width, height] = image.size
[block_size_x, block_size_y] = [math.floor(
    width/NUM_COL), math.floor(height/NUM_ROW)]
blocks = np.zeros((block_size_y, block_size_x, 3, NUM_COL * NUM_ROW))
image_copy = np.array(image)
for i in range(NUM_ROW):
    for j in range(NUM_COL):
        x = j * block_size_x
        y = i * block_size_y
        blocks[:, :, :, NUM_ROW * i + j] = image_copy[y:y +
                                                      block_size_y, x:x+block_size_x, :]

[suffle_image, block_images] = suffleImage(image, block_size_x, block_size_y)

image = np.array(image)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold = 0.85
swap_list = []
swapped = [0] * NUM_ROW * NUM_COL

for i in range(NUM_ROW*NUM_COL):
    template = block_images[:, :, :, i].astype('uint8')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    thresh_list = []
    for j in range(NUM_ROW*NUM_COL):
        if swapped[j] == 1:
            continue
        source_image = blocks[:, :, :, j].astype('uint8')
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(source_image, template, cv2.TM_CCOEFF_NORMED)
        if res >= threshold:
            thresh_list.append([j, res[0][0]])
    thresh_list = np.array(thresh_list)
    max_index = int(thresh_list[np.argmax(thresh_list[:, 1])][0])
    swap_list.append([i, max_index])
    swapped[max_index] = 1


block_images_solved = np.zeros(
    (block_size_y, block_size_x, 3, NUM_COL * NUM_ROW))
for swap in swap_list:
    block_images_solved[:, :, :, swap[1]] = block_images[:, :, :, swap[0]]
solved_image = np.zeros((height, width, 3), dtype='uint8')
for i in range(NUM_ROW):
    for j in range(NUM_COL):
        x = j * block_size_x
        y = i * block_size_y
        block_index = NUM_ROW * i + j
        solved_image[y:y+block_size_y, x:x+block_size_x,
                     ::-1] = block_images_solved[:, :, :, block_index]

cv2.imwrite('suffle_image.jpg', suffle_image)
cv2.imwrite('solved_image.jpg', solved_image)