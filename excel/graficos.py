import pandas as pd
import matplotlib.pyplot as plt

# Leer el primer archivo Excel
archivo_excel_1 = 'kernel_7_x_30.xlsx'
df1 = pd.read_excel(archivo_excel_1)

# Leer el segundo archivo Excel
archivo_excel_2 = 'kernel_7_y_31.xlsx'
df2 = pd.read_excel(archivo_excel_2)

# Asignar las columnas del primer archivo a variables
x1 = df1.iloc[:, 0]  # Primera columna
y1_1 = df1.iloc[:, 1]  # Segunda columna
y1_2 = df1.iloc[:, 2]  # Tercera columna
y1_3 = df1.iloc[:, 3]  # Cuarta columna

# Asignar las columnas del segundo archivo a variables
x2 = df2.iloc[:, 0]  # Primera columna
y2_1 = df2.iloc[:, 1]  # Segunda columna
y2_2 = df2.iloc[:, 2]  # Tercera columna
y2_3 = df2.iloc[:, 3]  # Cuarta columna

# Encontrar los mínimos del primer archivo
min_y1_1 = round(y1_1.min(), 2)
min_y1_2 = round(y1_2.min(), 2)
min_y1_3 = round(y1_3.min(), 2)

# Encontrar los mínimos del segundo archivo
min_y2_1 = round(y2_1.min(), 2)
min_y2_2 = round(y2_2.min(), 2)
min_y2_3 = round(y2_3.min(), 2)

# Crear subplots: 2 filas, 2 columnas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Primera gráfica: Archivo 1, Columna 2 y 3
axes[0, 0].plot(x1, y1_1, label='Train Error', color='blue')
axes[0, 0].plot(x1, y1_2, label='Validation Error', color='green')

# Resaltar los puntos mínimos en el gráfico de la columna 2 y 3 del primer archivo
axes[0, 0].scatter(x1[y1_2.idxmin()], min_y1_2, color='red', zorder=5, label=f'Mínimo Validation: {min_y1_2}')

# Configuración de la primera gráfica
axes[0, 0].set_xlabel('Épocas')
axes[0, 0].set_ylabel('Pixeles')
axes[0, 0].set_title('Error Promedio (Kernel 7 X)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Segunda gráfica: Archivo 1, Columna 4
axes[0, 1].plot(x1, y1_3, label='Train Loss', color='purple')

# Configuración de la segunda gráfica
axes[0, 1].set_xlabel('Épocas')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Pérdidas (Kernel 7 X)')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Tercera gráfica: Archivo 2, Columna 2 y 3
axes[1, 0].plot(x2, y2_1, label='Train Error', color='blue')
axes[1, 0].plot(x2, y2_2, label='Validation Error', color='green')

# Resaltar los puntos mínimos en el gráfico de la columna 2 y 3 del segundo archivo
axes[1, 0].scatter(x2[y2_2.idxmin()], min_y2_2, color='red', zorder=5, label=f'Mínimo Validación: {min_y2_2}')


# Configuración de la tercera gráfica
axes[1, 0].set_xlabel('Épocas')
axes[1, 0].set_ylabel('Pixeles')
axes[1, 0].set_title('Error Promedio (kernel 7 Y)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Cuarta gráfica: Archivo 2, Columna 4
axes[1, 1].plot(x2, y2_3, label='Train Loss', color='purple')

# Configuración de la cuarta gráfica
axes[1, 1].set_xlabel('Épocas')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Pérdidas (Kernel 7 Y)')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Ajustar el espaciado entre subplots
plt.tight_layout()

# Mostrar las cuatro gráficas
plt.show()
