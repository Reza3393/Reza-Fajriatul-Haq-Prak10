import matplotlib.pyplot as plt
import cv2
import mahotas.features.texture as mht

PATCH_SIZE = 21

# Memuat gambar menggunakan OpenCV
image = cv2.imread('Image/langit.jpg', cv2.IMREAD_GRAYSCALE)

grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    patch = image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE]
    if patch.size > 0:  # Memeriksa ukuran patch sebelum menambahkannya
        grass_patches.append(patch)

sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    patch = image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE]
    if patch.size > 0:  # Memeriksa ukuran patch sebelum menambahkannya
        sky_patches.append(patch)

# Menghitung GLCM
xs = []
ys = []
for patch in (grass_patches + sky_patches):
    glcm = mht.haralick(patch.astype(int))
    xs.append(glcm.mean(axis=0)[2])  # Dissimilarity
    ys.append(glcm.mean(axis=0)[4])  # Correlation

fig = plt.figure(figsize=(8, 8))

# Menampilkan gambar asli dengan lokasi patch
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Gambar Asli')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# Plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go', label='Rumput')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo', label='Langit')
ax.set_xlabel('Dissimilarity GLCM')
ax.set_ylabel('Korelasi GLCM')
ax.legend()

# Menampilkan patch rumput secara individual
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel('Rumput %d' % (i + 1))

# Menampilkan patch langit secara individual
for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel('Langit %d' % (i + 1))

# Menyesuaikan jarak dan menampilkan gambar
fig.suptitle('Fitur matriks co-occurrence level abu-abu', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
