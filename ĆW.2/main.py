import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Wczytanie własnego obrazu
image_path = 'image.webp'  # podaj ścieżkę do swojego obrazu
image = mpimg.imread(image_path)

# Jeśli obraz jest kolorowy, możesz przekonwertować go na skalę szarości
image_gray = np.mean(image, axis=2) if image.ndim == 3 else image

# Obliczenie macierzy korelacji dla wierszy (po każdym wierszu)
row_correlation = np.corrcoef(image_gray)

# Obliczenie macierzy korelacji dla kolumn (po każdej kolumnie)
column_correlation = np.corrcoef(image_gray.T)

# Wizualizacja macierzy korelacji
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(row_correlation, cmap='viridis', interpolation='nearest')
plt.title('Korelacja wierszy')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(column_correlation, cmap='viridis', interpolation='nearest')
plt.title('Korelacja kolumn')
plt.colorbar()

plt.tight_layout()
plt.show()
