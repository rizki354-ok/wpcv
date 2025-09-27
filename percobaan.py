import matplotlib.pyplot as plt

# Data sederhana
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Buat subplot 1 baris, 2 kolom
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Kolom 1: Grafik ---
ax1.plot(x, y, label="y = 2x", color="blue")
ax1.set_title("Grafik Linear")
ax1.set_xlabel("Sumbu X")
ax1.set_ylabel("Sumbu Y")
ax1.legend()

# Tambahkan anotasi
ax1.annotate("Nilai maksimum",
             xy=(5, 10), xycoords='data',
             xytext=(3.5, 9), textcoords='data',
             arrowprops=dict(arrowstyle="->", color="red"))

# --- Kolom 2: Deskripsi ---
deskripsi = (
    "Deskripsi Grafik:\n"
    "- Grafik menunjukkan hubungan linear y = 2x.\n"
    "- Setiap kenaikan 1 di X, nilai Y naik 2.\n"
    "- Titik maksimum dalam data ini ada di (5, 10).\n"
    "- Garis ini adalah garis lurus dengan gradien 2."
)


ax2.axis("on")  # hilangkan sumbu
ax2.text(0, 1, deskripsi, fontsize=11, va="top")

plt.tight_layout()
plt.show()
