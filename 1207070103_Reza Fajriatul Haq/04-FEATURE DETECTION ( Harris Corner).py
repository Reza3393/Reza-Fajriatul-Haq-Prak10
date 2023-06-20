import numpy as np
import cv2
from matplotlib import pyplot as plt

# Gunakan gambar
img = cv2.imread('Image/loco.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi pojok dengan metode Harris Corner Detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

# Ambil pojok-pojok yang cukup tajam (nilai dst melebihi ambang batas)
threshold = 0.01 * dst.max()
corner_img = np.copy(img)
corner_img[dst > threshold] = [0, 0, 255]  # Gambar titik-titik pojok berwarna merah

# Tampilkan jumlah titik pojok yang terdeteksi
corners = np.argwhere(dst > threshold)
num_corners = len(corners)
print("Jumlah titik terdeteksi:", num_corners)

# Tampilkan gambar dengan titik-titik pojok yang terdeteksi
plt.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
plt.title("Harris Corner Detection")
plt.axis('off')
plt.show()
