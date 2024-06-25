import numpy as np
import pandas as pd

# Wczytanie danych z pliku CSV z separatorem średnika
try:
    df = pd.read_csv('war9.csv', sep=';')
    print(df.head())  # opcjonalne: wyświetlenie pierwszych kilku wierszy, aby sprawdzić strukturę danych
except FileNotFoundError:
    print("Nie można znaleźć pliku dane.csv")
except Exception as e:
    print(f"Wystąpił problem podczas wczytywania danych: {e}")

# Konwersja kolumny 'y' na numeryczną, jeśli nie jest już numeryczna
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Wyodrębnienie kolumn x1, x2 i y jako numpy arrays
x1 = df['x1'].values
x2 = df['x2'].values
y = df['y'].values

# Tworzenie macierzy X z kolumn x1 i x2
X = np.column_stack((x1, x2))

# Obliczenie macierzy pseudoodwrotnej X+
X_plus = np.linalg.pinv(X)

# Obliczenie współczynników a i b
coefficients = np.dot(X_plus, y)

# Współczynniki a i b
a = coefficients[0]
b = coefficients[1]

print(f"Współczynnik a: {a}")
print(f"Współczynnik b: {b}")
