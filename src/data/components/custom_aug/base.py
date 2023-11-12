
import numpy as np
import cv2
import random
import math



################################### GLOBAL VARIABLES ##########################################
translator = dict()
################################### LOAD GLOBAL VARIABLES #####################################
def load_translator(dict_path):
    with open(dict_path, "r") as file:
        for line in file:
            part = line.strip().split()
            if len(part) > 1:
                translator[part[0]] = part[1]
    return True

def load_texture(path, intensity=1):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cant read image")
    bg_color = np.mean(img, axis=(0, 1))
    threshold = (img < bg_color).astype(np.uint8)
    cropped = crop_img(img / 255, threshold, 0, 0, 0, 0, bg_color=1)
    cropped = cropped ** intensity
    return cropped

#################################### PREPARE DATA ##################################################
#### LABEL
def translate(label):
    output = ""
    num_punc = 0
    for char in label:
        output += translator[char]
    
    for char in output:
        if char == '^' or char == '~' or char == '`' or char == '\'' or char == '?' or char == '.' or char == '_':
            num_punc += 1
    
    return output, len(output) - num_punc, num_punc

#### IMAGE
odd = lambda a: a + a % 2 - 1
def sharpen(img):
  # ksize must be odd
  h = 3
  w = 3
  sharpener = - np.ones((h, w))
  sharpener[int(h / 2), int(w / 2)] = h * w
  sharpened = cv2.blur(img, (odd(int(h/2)), odd(int(w/2))))
  sharpened = cv2.filter2D(sharpened, -1, sharpener)
  sharpened = cv2.bilateralFilter(sharpened, h * w, 150, 75)
  return sharpened

def smooth(img, color_dis, scale=75):
  # blurred = cv2.bilateralFilter(img, 21, color_dis, int(scale))
  # sharpened = sharpen(blurred)
  # return sharpened
  return img

def crop_img(img, mask, pad_bx, pad_by, pad_ex, pad_ey, bg_color, smoothing=False):
  if smoothing:
    img = smooth(img, 20)
  points = np.array(np.where(mask > 0)).T
  y, x, h, w = cv2.boundingRect(points)
  # plt.imshow(img)
  # plt.plot((x, x + w, x + w, x), (y, y, y + h, y + h))
#   plt.pause(0.2)
  bx = int(x - pad_bx)
  by = int(y - pad_by)
  ex = int(x + w + pad_ex)
  ey = int(y + h + pad_ey)
  ex_pad = int(max(ex - img.shape[1], 0))
  ey_pad = int(max(ey - img.shape[0], 0))
  bx_pad = - min(bx, 0)
  by_pad = - min(by, 0)
  # print(bx, ex, bx_pad, ex_pad)
  # print(by, ey, by_pad, ey_pad)
  cropped = img[max(0, by) : min(img.shape[0], ey), max(0, bx) : min(img.shape[1], ex)]
  padded = cv2.copyMakeBorder(cropped,
                              by_pad, ey_pad,
                              bx_pad, ex_pad,
                              borderType=cv2.BORDER_CONSTANT,
                              value=bg_color)
  return padded

# def crop_original(img, pad_bx, pad_by, pad_ex, pad_ey):
#   sub_img = smooth(img, 20, 75)
#   mask = sharp_mask(sub_img, 0.85, 0.85)

#   bg_mask = sub_img < (20, 20, 20)
#   bg_color = extract_median_color_axis(img.copy(), bg_mask, axis = (0, 1)).astype(np.uint8).tolist()
#   points = np.array(np.where(mask > 0)).T
#   y, x, h, w = cv2.boundingRect(points)
#   bx = int(x - pad_bx)
#   by = int(y - pad_by)
#   ex = int(x + w + pad_ex)
#   ey = int(y + h + pad_ey)
#   ex_pad = int(max(ex - img.shape[1], 0))
#   ey_pad = int(max(ey - img.shape[0], 0))
#   bx_pad = - min(bx, 0)
#   by_pad = - min(by, 0)
#   # print(bx, ex, bx_pad, ex_pad)
#   # print(by, ey, by_pad, ey_pad)
#   cropped = img[max(0, by) : min(img.shape[0], ey), max(0, bx) : min(img.shape[1], ex)]
#   padded = cv2.copyMakeBorder(cropped,
#                               by_pad, ey_pad,
#                               bx_pad, ex_pad,
#                               borderType=cv2.BORDER_CONSTANT,
#                               value=bg_color)
#   return padded
#### MASKING

# 1 cho chiều ngang
# 0 cho chiều dọc
def line_erosion(mask, axis = 0, threshold=0.9):
  dist = np.average(mask, axis=axis)
  # print(dist.max(), dist.shape)
  for i in range(dist.shape[0]):
    if dist[i] >= threshold:
      # print("Deleted at i-th")
      if axis == 0:
        mask[:, i] = 0
      else:
        mask[i, :] = 0
  return mask  

def extract_median_color_axis(img, threshold, axis = 0):
  color_threshold = threshold

  img[img < color_threshold * 255] = 0
  mean_color = np.mean(img, axis=axis)
  mean_dist = np.mean(color_threshold, axis = axis)
  background = mean_color / (1 - mean_dist)
  return background

def masking(img, grey=False, keep_line=True, line_x_erosion=0.9, line_y_erosion=0.9):
    sub_img = smooth(img, 20, 50)
    bg_color = np.mean(sub_img, axis=(0,1))
    mask = (sub_img < bg_color).astype(np.uint8)
    if grey is True:
        mask = np.max(mask, axis=2)
    
    h, w = img.shape[:2]
    final = None
    if keep_line is False:
        if h > w:
            final = line_erosion(mask, 1, line_x_erosion)
            final = line_erosion(final, 0, line_y_erosion)
        else:
            # print("Crop row first")
            final = line_erosion(mask, 0, line_y_erosion)
            final = line_erosion(final, 1, line_x_erosion)
    else:
        final = mask
    
    bg_color = extract_median_color_axis(sub_img, mask, axis=(0, 1))

    return final, bg_color


# load_translator("/work/hpc/firedogs/potato/translate.txt")
# load_texture("/work/hpc/firedogs/potato/asset/texture.png", intensity=2)
# print(texture)