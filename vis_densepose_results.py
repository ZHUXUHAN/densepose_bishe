import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

im  = cv2.imread('/home/bishe/densepose/vis_train_18641.png')
IUV = cv2.imread('/home/bishe/Pet-dev-2.11/vis_results/vis_train_18641_uv.png')
INDS = cv2.imread('/home/bishe/Pet-dev-2.11/vis_results/vis_train_18641_i_mask.png', 0)

fig = plt.figure(figsize=[12, 12])
plt.imshow(np.hstack((IUV[:, :, 0] / 24., IUV[:, :, 1] / 256., IUV[:, :, 2] / 256.)))
plt.title('I, U and V images.')
plt.axis('off')
plt.savefig("i.png")
plt.close()


def TransferTexture(TextureIm, im, IUV):
    U = IUV[:, :, 1]
    V = IUV[:, :, 2]
    #
    R_im = np.zeros(U.shape)
    G_im = np.zeros(U.shape)
    B_im = np.zeros(U.shape)
    ###
    for PartInd in range(1, 25):  ## Set to xrange(1,23) to ignore the face part.
        tex = TextureIm[PartInd - 1, :, :, :].squeeze()  # get texture for each part.
        #####
        R = tex[:, :, 0]
        G = tex[:, :, 1]
        B = tex[:, :, 2]
        ###############
        x, y = np.where(IUV[:, :, 0] == PartInd)
        u_current_points = U[x, y]  # Pixels that belong to this specific part.
        v_current_points = V[x, y]
        ##
        r_current_points = R[((255 - v_current_points) * 199. / 255.).astype(int), (
                    u_current_points * 199. / 255.).astype(int)] * 255
        g_current_points = G[((255 - v_current_points) * 199. / 255.).astype(int), (
                    u_current_points * 199. / 255.).astype(int)] * 255
        b_current_points = B[((255 - v_current_points) * 199. / 255.).astype(int), (
                    u_current_points * 199. / 255.).astype(int)] * 255
        ##  Get the RGB values from the texture images.
        R_im[IUV[:, :, 0] == PartInd] = r_current_points
        G_im[IUV[:, :, 0] == PartInd] = g_current_points
        B_im[IUV[:, :, 0] == PartInd] = b_current_points
    generated_image = np.concatenate((B_im[:, :, np.newaxis], G_im[:, :, np.newaxis], R_im[:, :, np.newaxis]),
                                     axis=2).astype(np.uint8)
    BG_MASK = generated_image == 0
    generated_image[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
    return generated_image


im_zero = np.zeros(IUV.shape)
##
Tex_Atlas = cv2.imread('./texture_from_SURREAL.png')[:, :, ::-1] / 255.
TextureIm = np.zeros([24, 200, 200, 3]);
for i in range(4):
    for j in range(6):
        TextureIm[(6 * i + j), :, :, :] = Tex_Atlas[(200 * j):(200 * j + 200), (200 * i):(200 * i + 200), :]
image = TransferTexture(TextureIm, im_zero, IUV)
plt.imshow(np.hstack((IUV, image[:, :, ::-1])))
plt.axis('off')
plt.savefig("b.png")
plt.close()

image = TransferTexture(TextureIm,im,IUV)
plt.imshow(np.hstack((im[:,:,::-1], image[:,:,::-1])))
plt.axis('off')
plt.savefig("c.png")
