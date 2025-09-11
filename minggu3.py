import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "apel.png"  # gambar default

def acara9_image_enhancement():
    img = cv2.imread(IMG_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # fix warna asli
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cerah = cv2.convertScaleAbs(gray, alpha=1, beta=50)
    gelap = cv2.convertScaleAbs(gray, alpha=1, beta=-50)
    neg = 255 - gray
    heq = cv2.equalizeHist(gray)

    # subplot citra grayscale + histogramnya
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(gray, cmap='gray')
    plt.title("Citra Grayscale")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.hist(gray.ravel(), bins=256, range=(0,256), color='black')
    plt.title("Histogram Grayscale")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.show()

    # tampilkan citra hasil enhancement lainnya
    titles = ['Asli', 'Cerah', 'Gelap', 'Negasi', 'Hist Eq']
    images = [img_rgb, cerah, gelap, neg, heq]  # gunakan img_rgb

    plt.figure(figsize=(12,6))
    for i in range(5):
        plt.subplot(2,3,i+1)
        cmap = 'gray' if len(images[i].shape) == 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def acara10_negasi():
    gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    neg = 255 - gray

    cv2.imshow("Grayscale", gray)
    cv2.imshow("Negasi", neg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def acara11_contrast_stretching():
    gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    min_val, max_val = np.min(gray), np.max(gray)
    cs = cv2.convertScaleAbs((gray - min_val) * (255.0 / (max_val - min_val)))

    cv2.imshow("Grayscale", gray)
    cv2.imshow("Contrast Stretching", cs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def acara12_histogram_equalization():
    img = cv2.imread(IMG_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heq = cv2.equalizeHist(gray)

    cv2.imshow("Asli", img)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Hist Equalization", heq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    while True:
        print("\n=== MENU PRAKTIKUM PENGOLAHAN CITRA (apel.png) ===")
        print("1. Acara 9 - Image Enhancement + Histogram")
        print("2. Acara 10 - Operasi Negasi")
        print("3. Acara 11 - Contrast Stretching")
        print("4. Acara 12 - Histogram Equalization")
        print("0. Keluar")
        pilihan = input("Pilih menu: ")

        if pilihan == "0":
            print("Program selesai.")
            break
        elif pilihan == "1":
            acara9_image_enhancement()
        elif pilihan == "2":
            acara10_negasi()
        elif pilihan == "3":
            acara11_contrast_stretching()
        elif pilihan == "4":
            acara12_histogram_equalization()
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()
