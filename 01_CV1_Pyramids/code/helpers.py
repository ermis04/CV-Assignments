import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

def blur_and_downsample(img, kernel_vec):
    h = kernel_vec
    rows, cols, channels = img.shape
    processed_channels = []

    for c in range(channels):
        channel = img[:, :, c]
        blurred = convolve2d(channel, h, mode='same', boundary='symm')
        blurred = convolve2d(blurred, h.T, mode='same', boundary='symm')
        downsampled = blurred[::2, ::2]
        processed_channels.append(downsampled)
    return np.stack(processed_channels, axis=-1)

def blur_and_upsample(img, kernel_vec):
    h = kernel_vec
    rows, cols, channels = img.shape
    upsampled_rows, upsampled_cols = rows * 2, cols * 2
    processed_channels = []

    for c in range(channels):
        channel = img[:, :, c]
        upsampled_channel = np.zeros((upsampled_rows, upsampled_cols))
        upsampled_channel[::2, ::2] = channel
        blurred = convolve2d(upsampled_channel, h, mode='same', boundary='symm')
        blurred = convolve2d(blurred, h.T, mode='same', boundary='symm')
        processed_channels.append(blurred)
    return np.stack(processed_channels, axis=-1)*4

def get_gaussian_pyramid(I, h, levels):
    gaussian_pyramid = [I]

    for i in range(1, levels):
        prev_img = gaussian_pyramid[i - 1]
        gaussian_pyramid.append(blur_and_downsample(prev_img, h))
    return gaussian_pyramid


def get_laplacian_pyramid(gaussian_pyramid, h, levels):
    laplacian_pyramid = []

    for i in range(levels-1):
        current_img = gaussian_pyramid[i]
        next_img = gaussian_pyramid[i + 1]
        expanded_img = blur_and_upsample(next_img, h)
        rows, cols = current_img.shape[:2]
        expanded_img = expanded_img[:rows, :cols, :]

        laplacian = current_img - expanded_img
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def show_pyramid(pyramid):
    import matplotlib.pyplot as plt

    levels = len(pyramid)
    fig, axes = plt.subplots(1, levels, figsize=(15, 5))

    for i in range(levels):
        axes[i].imshow(np.clip(pyramid[i], 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f'{pyramid[i].shape[0]}x{pyramid[i].shape[1]}')
    plt.show()

def blend(laplacianA, laplacianB, maskPyramid):
    blended_pyramid = []

    for i in range(len(laplacianA)):
        la = laplacianA[i]
        lb = laplacianB[i]
        mask = maskPyramid[i]

        blended = la * mask + lb * (1 - mask)
        blended_pyramid.append(blended)
    return blended_pyramid

def reconstruct_from_laplacian(laplacian_pyramid, h):
    curr_img = laplacian_pyramid[-1]
    reconstruction = [curr_img]

    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        expanded_img = blur_and_upsample(curr_img, h)
        laplacian = laplacian_pyramid[i]
        rows, cols = laplacian.shape[:2]
        expanded_img = expanded_img[:rows, :cols, :]
        curr_img = expanded_img + laplacian
        reconstruction.append(curr_img)
    return curr_img, reconstruction

def downsize_and_reposition(I1, I2, h, min_component=200, start_pos=(0,0)):
    small_enough = False
    while not small_enough:
        print("I1 shape:", I1.shape)
        if I2.shape[0] < min_component or I2.shape[1] < min_component:
            small_enough = True
        else:
            I2 = blur_and_downsample(I2, h)
    I3 = np.zeros((I1.shape[0], I1.shape[1], 3))

    rows_I1, cols_I1 = I1.shape[:2]
    rows_I2, cols_I2 = I2.shape[:2]
    start_r, start_c = start_pos
    end_r = min(start_r + rows_I2, rows_I1)
    end_c = min(start_c + cols_I2, cols_I1)
    h_fit = end_r - start_r
    w_fit = end_c - start_c

    if h_fit > 0 and w_fit > 0:
        I3[start_r:end_r, start_c:end_c, :] = I2[:h_fit, :w_fit, :]

    return I3

def generate_final_pyramid(I1, I2, h, pyramid_size, mask, show_pyramids=True):
    I1_gaussian = get_gaussian_pyramid(I1, h, pyramid_size)
    I1_laplacian = get_laplacian_pyramid(I1_gaussian, h, pyramid_size)
    I2_gaussian = get_gaussian_pyramid(I2, h, pyramid_size)
    I2_laplacian = get_laplacian_pyramid(I2_gaussian, h, pyramid_size)
    mask_gaussian = get_gaussian_pyramid(mask, h, pyramid_size)

    blend_pyramid = blend(I1_laplacian, I2_laplacian, mask_gaussian)
    final_img, reconstructed = reconstruct_from_laplacian(blend_pyramid, h)

    if show_pyramids:
        show_pyramid(I1_gaussian)
        show_pyramid(I1_laplacian)
        show_pyramid(I2_gaussian)
        show_pyramid(I2_laplacian)
        show_pyramid(mask_gaussian)
        show_pyramid(blend_pyramid)
        show_pyramid(reconstructed)

    plt.imshow(np.clip(final_img, 0, 1))
    return final_img, reconstructed
