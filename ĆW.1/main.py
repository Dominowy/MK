# Importowanie bibliotek
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

# Ustawienie rozmiarów wykresów
plt.rcParams['figure.figsize'] = [16, 8]

# Wczytanie obrazu i konwersja do skali szarości
A = imread('image.webp')
X = np.mean(A, -1)  # Konwersja obrazu RGB do skali szarości

# Wyświetlenie oryginalnego obrazu w skali szarości
img = plt.imshow(256 - X)
img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# Dekompozycja SVD obrazu
U, S, VT = np.linalg.svd(X, full_matrices=False)
print(S.shape)
S = np.diag(S)

# Wyświetlenie przybliżonych obrazów przy różnych liczbach wartości singularnych
j = 0
for r in (5, 20, 100, 650):
    # Konstrukcja przybliżonego obrazu
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    
    # Wyświetlenie przybliżonego obrazu
    plt.figure(j + 1)
    j += 1
    img = plt.imshow(256 - Xapprox)
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r=' + str(r))
    plt.show()

# Wykres wartości singularnych w skali logarytmicznej
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

# Wykres skumulowanej sumy wartości singularnych
plt.figure(2)
plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.show()
