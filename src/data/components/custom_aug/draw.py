import numpy as np
import cv2
import random
import math

from src.data.components.custom_aug.base import *

# texture = load_texture("/work/hpc/firedogs/potato/asset/texture.png", intensity=2)
#################### CHARACTER AND PUNCTUATION RECOGNITION ############################
def find_centroid(mask, bboxes):
  centroid = []
  for i in range(len(bboxes)):
    x, y, w, h = bboxes[i]
    center = np.median(np.where(mask[y:y+h, x:x+w] > 0), axis=1) + [y, x]
    # print(center)
    centroid.append(center)
  # plt.imshow(img)
  return centroid

def regression(points2d):
  x = np.ones((points2d.shape[0], 2))
  x[:, 0] = points2d[:, 1]
  param = np.linalg.lstsq(x, points2d[:, 0], rcond=None)[0]
  return param

def find_text_and_punc(img, mask = None, label=""):
  if mask is None:
    sub_img = smooth(img, 20, 50)
    bg_color = np.mean(sub_img, axis=(0, 1))
    # print("Background filter: ", bg_color)
    mask = np.max((sub_img < bg_color).astype(int), axis = 2).astype(np.uint8)

  cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = np.array(cnts[len(cnts) % 2])
  # print(cnts.shape, " Contours", cnts[0].shape)
  
  punc = 0
  for char in label:
    if char == '^' or char == '~' or char == '`' or char == '\'' or char == '?' or char == '.' or char == '_':
      punc += 1
  num_char = len(label) - punc
  
  ratio = np.sum(mask) / (num_char * 1.2 + punc * 0.5)
  # print("Threshold: ", ratio)

  boxes = np.array([cv2.boundingRect(cnt) for cnt in cnts])
  # print(boxes.shape[0])

  is_char = np.full((len(cnts),), False, dtype=bool)
  for i in range(boxes.shape[0]):
    x, y, w, h = boxes[i]
    convex = cv2.convexHull(cnts[i])
    # print(convex)
    area = cv2.contourArea(convex)

    if area > 0.4 * h * w and ((h >= img.shape[0] * 0.1) or (w >= img.shape[1] * 0.1)):
      if np.sum(mask[y: y + h, x: x + w]) > ratio :
        is_char[i] = True
        # print("Its a char")
      else:
        min_rect = cv2.minAreaRect(cnts[i])
        if min_rect[2] < 10 and min_rect[1][0] * 2 < min_rect[1][1]:
          is_char[i] = True
          # print("Straight-up char")
        else:
          is_char[i]= False
          # print("Not promising")
    else:
      # print(area)
      is_char[i] = False
      # print("Not filling")
  # print(is_char)
  # print(boxes[is_char == True].shape)

  centroid = None
  centroid = np.array(find_centroid(mask, boxes))

  if np.sum(is_char) > 1:
    param = regression(centroid[is_char == True])
    # print(param)
    max_dist = np.max(np.abs(np.sum(centroid[is_char == True] * [-1, param[0]], axis=1) + param[1]) / np.sqrt(param[0] * param[0] + 1))
    punc_dist = np.abs(np.sum(centroid[is_char == False] * [-1, param[0]], axis=1) + param[1]) / np.sqrt(param[0] * param[0] + 1)
    punc_loc = np.where(is_char == False)[0]
    # print(punc_loc)
    # print(punc_dist, " Maximum:", max_dist)
    for i in range(len(punc_loc)):
        if punc_dist[i] < max(max(img.shape[0], img.shape[1]) / 10, max_dist):
          is_char[punc_loc[i]] = True

  # print("There are {0} text and {1} punctuation".format(np.sum(is_char.astype(int)), len(label) - np.sum(is_char.astype(int))))
  return cnts, boxes, is_char, centroid

def line_vector(cnts,
                boxes,
                is_char,
                centroid,
                img_size,
                y_noise = 0,
                skew_noise = 0,
                intent = False,
                align = 0,
                ignore_skew = False,
                axis = 1):
  a, b = 0, 0
  if axis == 0:
    ignore_skew = True
  if np.sum((is_char == False).astype(int)) == 0:
    intent = False
    
  if intent is True:
    punc_loc = np.array(np.where(is_char == False))
    index = np.random.choice(punc_loc.flatten())
    if axis == 0:
      a = np.random.normal(0, skew_noise)
      b = centroid[index, 1] - centroid[index, 0] * a
    else:
      a = np.random.normal(0, skew_noise)
      b = centroid[index, 0] - centroid[index, 1] * a
    return a, b

  else:
    a, b = regression(centroid[is_char == True])
    bound = np.mean(np.min(boxes[is_char == True, 2:4], axis = 1))
    b += bound * align
    b += np.random.normal(0, y_noise)
    if ignore_skew is True:
      x = np.random.randint(0, img_size[1])
      y = a * x + b
      if axis == 0:
        A = np.random.normal(0, skew_noise)
        B = x - y * a
      else:
        A = np.random.normal(0, skew_noise)
        B = y - x * a
      return A, B
    else:
      a += np.random.normal(0, skew_noise)
      return a, b

def draw_line(img, pattern, spacing, A, b, axis=1, noise_level=0):
  # print(img.shape, pattern.shape)
  if axis == 1:
    x = np.arange(0, img.shape[1], spacing, dtype=int)

    y = (x * A + b).astype(int)
    # print(y)
  else:
    y = np.arange(0, img.shape[0], spacing, dtype=int)
    x = (y * A + b).astype(int)
    # print(x)
  mask = np.ones(img.shape[0:2], dtype = float)
  noise = np.random.normal(1, noise_level, mask.shape)
  # print(noise)
  for i in range(x.shape[0]):
    if x[i] < 0 or y[i] < 0:
      continue
    shape = mask[y[i]:y[i] + pattern.shape[0], x[i]:x[i] + pattern.shape[1]].shape
    # if shape < pattern.shape:
    #   print(x[i], y[i], " failed")
    mask[y[i]:y[i] + shape[0], x[i]:x[i] + shape[1]] = pattern[:shape[0], :shape[1]]

  mask = cv2.GaussianBlur(mask, (3, 3), 0)
  mask *= noise
  output = img * np.stack((mask, mask, mask), axis = 2)
  return output.astype(int).astype(np.uint8)


##### WRAPPER 
def line_and_noise(img, label, mask=None, bg_color = 1, line_width = 2, y_noise=0,skew_noise=0, intent=False, align=0, ignore_skew=False, axis=1, spacing=0, noise_level=0):
    cnts, boxes, is_char, centroid = find_text_and_punc(img, mask, label)

    a, b = line_vector(cnts, boxes, is_char, centroid, img.shape, y_noise, skew_noise, intent=intent, align=align, ignore_skew=ignore_skew, axis=axis)
    customized_pattern = cv2.resize(texture, (line_width, line_width), interpolation=cv2.INTER_LINEAR)
    output = draw_line(img, customized_pattern, spacing, a, b, axis=axis, noise_level=noise_level)
    return output
 
#### TEST
if __name__ =="__main__":
    img = cv2.imread("/work/hpc/firedogs/potato/check/img_original.jpg")
    output = line_and_noise(img, "chộn", mask=None,
                                        bg_color=None,
                                        line_width=2,
                                        y_noise=4,
                                        intent=True,
                                        align=-1,
                                        ignore_skew=False,
                                        axis=1,
                                        spacing=2,
                                        noise_level=0.2)
    print(output.shape, output.dtype)
    cv2.imwrite("/work/hpc/firedogs/data_/new_train/train_img_{0}.{1}".format(2445, "jpg"), output)