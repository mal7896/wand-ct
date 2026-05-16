import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform

# replace path file with your actual ct scan
image = io.imread(
    r"C:\Users\anonymous_user\data.png",
    as_gray=True
)

image = image / 255.0

wavelet = 'db3'
decomp_level = 7

lf_images = []
hf_images = []

orig_size = image.shape

coeffs = pywt.wavedec2(image, wavelet, level=decomp_level)

# LF subband images
for i in range(1, decomp_level + 1):
    lf_coeffs = pywt.wavedec2(image, wavelet, level=i)
    lf_subband = lf_coeffs[0]

    upscaled_lf = transform.resize(
        lf_subband,
        orig_size,
        order=3,
        mode='reflect',
        anti_aliasing=True
    )

    lf_images.append(upscaled_lf)

# HF subband images
for j in range(0, decomp_level):

    if j == 0:
        hf_subbands = pywt.idwt2(
            [np.zeros_like(coeffs[0]), coeffs[1]],
            wavelet
        )

        upscaled_hf = transform.resize(
            hf_subbands,
            orig_size,
            order=3,
            mode='reflect',
            anti_aliasing=True
        )

        hf_images.append(upscaled_hf)

    else:
        hf_coeffs = pywt.wavedec2(
            image,
            wavelet,
            level=(decomp_level - j)
        )

        a = 0

        hf_coeffs[1] = list(hf_coeffs[1])

        hf_coeffs_mod = [
            [
                [0 for l in range(len(hf_coeffs[1][0]))]
                for m in range(len(hf_coeffs[1][0]) + 1)
            ]
            for n in range(3)
        ]

        hf_coeffs_mod_2 = [
            [
                [0 for l in range(len(hf_coeffs[1][0] + 1))]
                for m in range(len(hf_coeffs[1][0]) + 2)
            ]
            for n in range(3)
        ]

        for k in range(0, len(hf_subbands) - len(hf_coeffs[1][0])):

            if k == 0:
                a = 1

                pad = [[0] * len(hf_coeffs[1][0])]

                hf_coeffs[1][0] = np.concatenate(
                    (hf_coeffs[1][0], pad)
                )

                hf_coeffs[1][1] = np.concatenate(
                    (hf_coeffs[1][1], pad)
                )

                hf_coeffs[1][2] = np.concatenate(
                    (hf_coeffs[1][2], pad)
                )

                n = 0

                for i in range(0, 3):
                    for j in range(0, len(hf_coeffs[1][0])):
                        n += 1

                        hf_coeffs_mod[i][j] = np.append(
                            hf_coeffs[1][i][j],
                            0
                        )

                        hf_coeffs_mod[i][j] = np.array(
                            hf_coeffs_mod[i][j]
                        )

            else:
                a = 2

                pad = [[0] * len(hf_coeffs[1][0])]

                hf_coeffs_mod[0] = np.concatenate(
                    (hf_coeffs_mod[0], pad)
                )

                hf_coeffs_mod[1] = np.concatenate(
                    (hf_coeffs_mod[1], pad)
                )

                hf_coeffs_mod[2] = np.concatenate(
                    (hf_coeffs_mod[2], pad)
                )

                n = 0

                for i in range(0, 3):
                    for j in range(0, len(hf_coeffs_mod[0])):
                        n += 1

                        hf_coeffs_mod_2[i][j] = np.append(
                            hf_coeffs_mod[i][j],
                            0
                        )

                        hf_coeffs_mod_2[i][j] = np.array(
                            hf_coeffs_mod_2[i][j]
                        )

        if a == 0:
            hf_subbands = pywt.idwt2(
                [hf_subbands, hf_coeffs[1]],
                wavelet
            )

        elif a == 1:
            hf_subbands = pywt.idwt2(
                [hf_subbands, tuple(hf_coeffs_mod)],
                wavelet
            )

        else:
            hf_subbands = pywt.idwt2(
                [hf_subbands, tuple(hf_coeffs_mod_2)],
                wavelet
            )

        upscaled_hf = transform.resize(
            hf_subbands,
            orig_size,
            order=3,
            mode='reflect',
            anti_aliasing=True
        )

        hf_images.append(upscaled_hf)

fig, axes = plt.subplots(2, decomp_level, figsize=(20, 10))

for i, img in enumerate(lf_images):
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].set_title(f'Low-Frequency Level {i + 1}')
    axes[0, i].axis('off')

for i, img in enumerate(hf_images):
    norm_img = (
        (img - np.min(img)) /
        (np.max(img) - np.min(img))
    )

    axes[1, i].imshow(norm_img, cmap='gray')
    axes[1, i].set_title(f'High-Frequency Level {i + 1}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
