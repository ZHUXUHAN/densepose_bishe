from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from random import randint
import matplotlib as mpl

mpl.use('Agg')

coco_folder = '/train/trainset/1/mscoco2014/'
dp_coco = COCO(coco_folder + '/annotations/densepose_coco_2014_train.json')

im_ids = dp_coco.getImgIds()
# Selected_im = im_ids[randint(0, len(im_ids))]
# print(Selected_im)
Selected_im = 18641
im = dp_coco.loadImgs(Selected_im)[0]
ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
anns = dp_coco.loadAnns(ann_ids)
im_name = os.path.join(coco_folder + 'train2014', im['file_name'])
I = cv2.imread(im_name)
fig, ax = plt.subplots()

plt.imshow(I[:, :, ::-1])
plt.axis('off')
height, width, channels = I.shape
# 如果dpi=300，那么图像大小=height*width
fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)

plt.savefig('vis_train_{}_tmp.png'.format(str(Selected_im)))
plt.close()

plt.imshow(I[:, :, ::-1])
plt.axis('off')
height, width, channels = I.shape
# 如果dpi=300，那么图像大小=height*width
fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)

plt.savefig('vis_train_{}.png'.format(str(Selected_im)))
plt.close()


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256, 256])
    for i in range(1, 15):
        if (Polys[i - 1]):
            current_mask = mask_util.decode(Polys[i - 1])
            MaskGen[current_mask > 0] = i
    return MaskGen


def GetDensePoseMask_i(Polys, i):
    if Polys[i - 1]:
        current_mask = mask_util.decode(Polys[i - 1])
        return current_mask
    else:
        return np.zeros([256, 256])


I_vis = I.copy()

for ann in anns:
    bbr = np.array(ann['bbox']).astype(int)  # the box.
    if 'dp_masks' in ann.keys():  # If we have densepose annotation for this ann,
        ################
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        cv2.rectangle(I_vis, (x1, y1), (x2, y2), (0, 0, 155), 2)

img_alpha = I_vis[:, :, ::-1]
plt.imshow(img_alpha)
plt.axis('off')
height, width, channels = img_alpha.shape
# 如果dpi=300，那么图像大小=height*width
fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('vis_train_{}_box.png'.format(str(Selected_im)))
plt.close()


for ann in anns:
    bbr = np.array(ann['bbox']).astype(int)  # the box.
    if 'dp_masks' in ann.keys():  # If we have densepose annotation for this ann,
        Mask = GetDensePoseMask(ann['dp_masks'])

        ################
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        x2 = min([x2, I.shape[1]]);
        y2 = min([y2, I.shape[0]])
        ###############
        MaskIm = cv2.resize(Mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)

        MaskBool = np.tile((MaskIm == 0)[:, :, np.newaxis], [1, 1, 3])
        #  Replace the visualized mask image with I_vis.
        Mask_vis = cv2.applyColorMap((MaskIm * 15).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
        Mask_vis[MaskBool] = I_vis[y1:y2, x1:x2, :][MaskBool]
        I_vis[y1:y2, x1:x2, :] = I_vis[y1:y2, x1:x2, :] * 0.7 + Mask_vis * 0.3

for ann in anns:
    bbr = np.array(ann['bbox']).astype(int)  # the box.
    if 'dp_masks' in ann.keys():  # If we have densepose annotation for this ann,
        ################
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        x2 = min([x2, I.shape[1]]);
        y2 = min([y2, I.shape[0]])
        ###############
        for i in range(1, 15):
            Mask = GetDensePoseMask_i(ann['dp_masks'], i)
            MaskIm = cv2.resize(Mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
            _, contours, _ = cv2.findContours(MaskIm.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(I_vis[y1:y2, x1:x2, :], contours, -1, (0, 0, 0), 3)

from PIL import Image


def addTransparency(img, factor=0.7):
    img = Image.fromarray(img, 'RGB')
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img_blender, img, factor)
    return img


def gray(img):
    img = Image.fromarray(img, 'RGB')
    im_gray = img.convert('L')
    # im2 = (100.0/255)*im_gray +100
    return im_gray


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# gray = rgb2gray(I_vis[:, :, ::-1])
# img_alpha = gray(I_vis[:, :, ::-1])
# img_alpha = np.array(img_alpha)
# img_gray = cv2.cvtColor(I_vis[:, :, ::-1],cv2.COLOR_RGB2GRAY)
# img_alpha = (100.0/255)*img_gray +100
# img_alpha = addTransparency(I_vis[:, :, ::-1])
# img_alpha = np.array(img_alpha)
img_alpha = I_vis[:, :, ::-1]

plt.imshow(img_alpha)
plt.axis('off')
height, width, channels = img_alpha.shape
# 如果dpi=300，那么图像大小=height*width
fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)

plt.savefig('vis_train_{}_parsing.png'.format(str(Selected_im)))
plt.close()

# fig = plt.figure(figsize=[15, 5])
# plt.subplot(1, 3, 1)
# plt.imshow(img_alpha)
# plt.axis('off')
# plt.title('Patch Indices')
# plt.subplot(1, 3, 2)
# plt.imshow(img_alpha)
# plt.axis('off')
# plt.title('U coordinates')
# plt.subplot(1, 3, 3)
# plt.imshow(img_alpha)
# plt.axis('off')
# plt.title('V coordinates')

## For each ann, scatter plot the collected points.
for ann in anns:
    bbr = np.round(ann['bbox'])
    if ('dp_masks' in ann.keys()):
        Point_x = np.array(ann['dp_x']) / 255. * bbr[2]  # Strech the points to current box.
        Point_y = np.array(ann['dp_y']) / 255. * bbr[3]  # Strech the points to current box.
        #
        Point_I = np.array(ann['dp_I'])
        Point_U = np.array(ann['dp_U'])
        Point_V = np.array(ann['dp_V'])
        #
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        x2 = min([x2, img_alpha.shape[1]]);
        y2 = min([y2, img_alpha.shape[0]])
        ###############
        Point_x = Point_x + x1
        Point_y = Point_y + y1
        #######绘制散点图#######
        plt.scatter(Point_x, Point_y, 6, Point_I)

# cv2.imwrite('vis_train_{}_i.png'.format(str(Selected_im)), img_alpha[:, :, ::-1])
plt.imshow(img_alpha)
plt.axis('off')

height, width, channels = img_alpha.shape
# 如果dpi=300，那么图像大小=height*width
fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('vis_train_{}_i.png'.format(str(Selected_im)))
plt.close()

for ann in anns:
    bbr = np.round(ann['bbox'])
    if ('dp_masks' in ann.keys()):
        Point_x = np.array(ann['dp_x']) / 255. * bbr[2]  # Strech the points to current box.
        Point_y = np.array(ann['dp_y']) / 255. * bbr[3]  # Strech the points to current box.
        #
        Point_I = np.array(ann['dp_I'])
        Point_U = np.array(ann['dp_U'])
        Point_V = np.array(ann['dp_V'])
        #
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        x2 = min([x2, img_alpha.shape[1]]);
        y2 = min([y2, img_alpha.shape[0]])
        ###############
        Point_x = Point_x + x1
        Point_y = Point_y + y1
        plt.scatter(Point_x, Point_y, 6, Point_V)

plt.imshow(img_alpha)
plt.axis('off')
height, width, channels = img_alpha.shape
# 如果dpi=300，那么图像大小=height*width
fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)

plt.savefig('vis_train_{}_u.png'.format(str(Selected_im)))
plt.close()

for ann in anns:
    bbr = np.round(ann['bbox'])
    if ('dp_masks' in ann.keys()):
        Point_x = np.array(ann['dp_x']) / 255. * bbr[2]  # Strech the points to current box.
        Point_y = np.array(ann['dp_y']) / 255. * bbr[3]  # Strech the points to current box.
        #
        Point_I = np.array(ann['dp_I'])
        Point_U = np.array(ann['dp_U'])
        Point_V = np.array(ann['dp_V'])
        #
        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        x2 = min([x2, img_alpha.shape[1]]);
        y2 = min([y2, img_alpha.shape[0]])
        ###############
        Point_x = Point_x + x1
        Point_y = Point_y + y1
        plt.scatter(Point_x, Point_y, 6, Point_U)

plt.imshow(img_alpha)
plt.axis('off')
height, width, channels = img_alpha.shape
# 如果dpi=300，那么图像大小=height*width
fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)

plt.savefig('vis_train_{}_v.png'.format(str(Selected_im)))
plt.close()
