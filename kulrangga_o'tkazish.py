import cv2
import matplotlib.pyplot as plt

# 1. Tasvirni o‘qish
img = cv2.imread('rasm.jpg')

# 2. Kulrangga o‘tkazish
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Oddiy threshold (T=128)
_, binary_simple = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# 4. Otsu usuli
_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Adaptiv usul
binary_adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# 6. Natijani ko‘rsatish
plt.figure(figsize=(12,6))
plt.subplot(1,4,1), plt.imshow(gray, cmap='gray'), plt.title('Kulrang')
plt.subplot(1,4,2), plt.imshow(binary_simple, cmap='gray'), plt.title('Oddiy Threshold')
plt.subplot(1,4,3), plt.imshow(binary_otsu, cmap='gray'), plt.title('Otsu usuli')
plt.subplot(1,4,4), plt.imshow(binary_adaptive, cmap='gray'), plt.title('Adaptiv usul')
plt.show()
