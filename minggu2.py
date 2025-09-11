import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# === Setup folder output ===
OUTPUT_DIR = "hasil"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download(filename, image):
    """Simpan hasil operasi ke folder 'hasil'."""
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, image)
    print(f"[OK] Hasil disimpan di {path}")

def show(title, imgs):
    n = len(imgs)
    plt.figure(figsize=(12,4))
    for i, (label, img) in enumerate(imgs):
        plt.subplot(1, n, i+1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(label)
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

# === Load citra ===
img = cv2.imread("apel.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
L = 256

# === Operasi dengan auto save ===
def grayscale():
    show("Grayscale", [("Asli", img), ("Grayscale", gray)])
    download("grayscale.png", gray)

def negatif():
    neg = (L-1) - gray
    show("Negatif", [("Grayscale", gray), ("Negatif", neg)])
    download("negatif.png", neg)

def clipping():
    def nothing(x): pass
    window = "Clipping"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    # buat trackbar
    cv2.createTrackbar("a", window, 0, 255, nothing)
    cv2.createTrackbar("b", window, 255, 255, nothing)

    # tampilkan gambar awal supaya window beneran terbuka
    cv2.imshow(window, gray)
    cv2.waitKey(500)  # kasih waktu GUI aktif

    while True:
        # cek apakah window masih ada
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return

        a = cv2.getTrackbarPos("a", window)
        b = cv2.getTrackbarPos("b", window)
        clipped = np.clip(gray, a, b)

        cv2.imshow(window, clipped)

        key = cv2.waitKey(30) & 0xFF
        if key in [13, 32]:  # Enter/Space
            break
        elif key == 27:      # Esc
            cv2.destroyWindow(window)
            return

    cv2.destroyWindow(window)
    show("Clipping (Final)", [("Asli", gray), (f"Clipped {a}-{b}", clipped)])
    download(f"clipping_{a}_{b}.png", clipped)


def penjumlahan():
    def nothing(x): pass
    window = "Penjumlahan"
    cv2.namedWindow(window)
    cv2.createTrackbar("Const", window, 0, 100, nothing)
    cv2.imshow(window, gray)
    cv2.waitKey(100)

    while True:
        k = cv2.getTrackbarPos("Const", window)
        C = np.clip(gray.astype(int) + k, 0, 255).astype(np.uint8)
        cv2.imshow(window, C)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]: break
        elif key == 27: return
    cv2.destroyWindow(window)
    show("Penjumlahan (Final)", [("Asli", gray), (f"+{k}", C)])
    download(f"penjumlahan_{k}.png", C)

def pengurangan():
    def nothing(x): pass
    window = "Pengurangan"
    cv2.namedWindow(window)
    cv2.createTrackbar("Const", window, 0, 100, nothing)
    cv2.imshow(window, gray)
    cv2.waitKey(100)

    while True:
        k = cv2.getTrackbarPos("Const", window)
        C = np.clip(gray.astype(int) - k, 0, 255).astype(np.uint8)
        cv2.imshow(window, C)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]: break
        elif key == 27: return
    cv2.destroyWindow(window)
    show("Pengurangan (Final)", [("Asli", gray), (f"-{k}", C)])
    download(f"pengurangan_{k}.png", C)

def perkalian():
    def nothing(x): pass
    window = "Perkalian"
    cv2.namedWindow(window)
    cv2.createTrackbar("Const", window, 10, 30, nothing)  # 0.1 - 3.0
    cv2.imshow(window, gray)
    cv2.waitKey(100)

    while True:
        k = cv2.getTrackbarPos("Const", window) / 10
        C = np.clip(gray.astype(float) * k, 0, 255).astype(np.uint8)
        cv2.imshow(window, C)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]: break
        elif key == 27: return
    cv2.destroyWindow(window)
    show("Perkalian (Final)", [("Asli", gray), (f"x{k}", C)])
    download(f"perkalian_{k:.1f}.png", C)

def pembagian():
    def nothing(x): pass
    window = "Pembagian"
    cv2.namedWindow(window)
    cv2.createTrackbar("Const", window, 1, 20, nothing)
    cv2.imshow(window, gray)
    cv2.waitKey(100)

    while True:
        k = cv2.getTrackbarPos("Const", window)
        if k == 0: k = 1
        C = np.clip(gray.astype(float) / k, 0, 255).astype(np.uint8)
        cv2.imshow(window, C)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]: break
        elif key == 27: return
    cv2.destroyWindow(window)
    show("Pembagian (Final)", [("Asli", gray), (f"/{k}", C)])
    download(f"pembagian_{k}.png", C)

def operasi_and():
    mask = np.zeros_like(gray)
    cv2.circle(mask, (gray.shape[1]//2, gray.shape[0]//2), 100, 255, -1)
    C = np.bitwise_and(gray, mask)
    show("AND", [("Asli", gray), ("Mask", mask), ("AND", C)])
    download("and.png", C)

def operasi_or():
    mask = np.zeros_like(gray)
    pts = np.array([[50,50],[200,100],[100,300]], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    C = np.bitwise_or(gray, mask)
    show("OR", [("Asli", gray), ("Mask", mask), ("OR", C)])
    download("or.png", C)

def operasi_xor():
    mask = np.zeros_like(gray)
    step = 40
    for y in range(0, mask.shape[0], step*2):
        mask[y:y+step, :] = 255
    C = np.bitwise_xor(gray, mask)
    show("XOR", [("Asli", gray), ("Mask", mask), ("XOR", C)])
    download("xor.png", C)

def translasi():
    def nothing(x): pass
    rows, cols = gray.shape
    window = "Translasi"
    cv2.namedWindow(window)
    cv2.createTrackbar("Tx", window, 0, cols, nothing)
    cv2.createTrackbar("Ty", window, 0, rows, nothing)
    cv2.imshow(window, gray)
    cv2.waitKey(100)

    while True:
        Tx = cv2.getTrackbarPos("Tx", window)
        Ty = cv2.getTrackbarPos("Ty", window)
        M = np.float32([[1,0,Tx],[0,1,Ty]])
        trans = cv2.warpAffine(gray, M, (cols, rows))
        cv2.imshow(window, trans)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]: break
        elif key == 27: return
    cv2.destroyWindow(window)
    show("Translasi (Final)", [("Asli", gray), (f"Geser {Tx},{Ty}", trans)])
    download(f"translasi_{Tx}_{Ty}.png", trans)

def cropping():
    clone = img.copy()
    r = cv2.selectROI("Pilih area cropping", clone, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Pilih area cropping")
    x, y, w, h = r
    if w > 0 and h > 0:
        crop = gray[y:y+h, x:x+w]
        show("Cropping", [("Asli", gray), ("Crop", crop)])
        download(f"cropping_{x}_{y}_{w}_{h}.png", crop)
    else:
        print("Tidak ada area dipilih.")

def flipping():
    flip_h = np.fliplr(gray)
    flip_v = np.flipud(gray)
    show("Flipping", [("Asli", gray), ("Horizontal", flip_h), ("Vertical", flip_v)])
    download("flip_horizontal.png", flip_h)
    download("flip_vertical.png", flip_v)

# === Menu interaktif ===
opsi = {
    "1": grayscale,
    "2": negatif,
    "3": clipping,
    "4": penjumlahan,
    "5": pengurangan,
    "6": perkalian,
    "7": pembagian,
    "8": operasi_and,
    "9": operasi_or,
    "10": operasi_xor,
    "11": translasi,
    "12": cropping,
    "13": flipping
}

while True:
    print("\n=== MENU OPERASI CITRA (SEMI OTOMATIS + AUTO SAVE) ===")
    print("1. Grayscale")
    print("2. Negatif")
    print("3. Clipping (slider)")
    print("4. Penjumlahan (slider)")
    print("5. Pengurangan (slider)")
    print("6. Perkalian (slider)")
    print("7. Pembagian (slider)")
    print("8. AND (mask)")
    print("9. OR (mask)")
    print("10. XOR (mask)")
    print("11. Translasi (slider)")
    print("12. Cropping (ROI)")
    print("13. Flipping")
    print("0. Keluar")

    pilihan = input("Pilih operasi (0-13): ")

    if pilihan == "0":
        print("Selesai.")
        break
    elif pilihan in opsi:
        opsi[pilihan]()
    else:
        print("Pilihan tidak valid!")