import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Podatki in normalizacija
X_original = np.array([10, 15, 20, 25, 30])     # Temperature v °C
X = X_original / max(X_original)                # Normalizacija glede na maksimalno vrednost
y = np.array([20, 40, 60, 80, 100])             # Število najemov koles

# Gradientni spust - konfiguracija
a, b = 0.0, 0.0
learning_rate = 0.1
epochs = 10000

# Nastavitve za vizualizacijo
selected_epochs = [0, 1, 2, 5, 10, 50, 100, 200, 1000, 9999] # Izberemo ključne iteracije za prikaz
colors = cm.viridis(np.linspace(0, 1, len(selected_epochs))) # Barvni gradient

# Priprava grafa
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot za originalne podatke
ax.scatter(X_original, y, color='blue', zorder=3, label='Dejanski podatki', s=80)

# Shranjevanje zgodovine
history = []

# Gradientni spust
for epoch in range(epochs):
    y_pred = a * X + b
    error = y_pred - y
    
    # Računanje gradientov
    grad_a = (2/len(X)) * np.dot(X, error)
    grad_b = (2/len(X)) * np.sum(error)
    
    # Posodabljanje parametrov
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    
    # Shranjevanje izbranih iteracij
    if epoch in selected_epochs:
        history.append((a.copy(), b.copy(), epoch+1))

# Barvna lestvica
norm = Normalize(vmin=0, vmax=len(selected_epochs)-1)
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

# Risanje regresijskih premic
for idx, ((a_val, b_val, iteracija), color) in enumerate(zip(history, colors)):
    x_line = np.linspace(min(X_original), max(X_original), 100)
    x_line_normalized = x_line / max(X_original)
    y_line = a_val * x_line_normalized + b_val
    
    ax.plot(x_line, y_line, color=color, alpha=0.8, linewidth=2.5, 
            label=f'Iter {iteracija} (y={a_val/max(X_original):.2f}x + {b_val:.2f})')

# Nastavitve grafa
ax.set_title('Razvoj linearne regresije skozi iteracije', fontsize=14, pad=20)
ax.set_xlabel('Temperatura (°C)', fontsize=12)
ax.set_ylabel('Število najemov', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(title='Iteracije', fontsize=9, title_fontsize=10, loc='upper left')

# Dodajanje barvne lestvice
cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(0, len(selected_epochs)-1, len(selected_epochs)))
cbar.set_label('Napredovanje učenja', fontsize=10)
cbar.set_ticklabels([f'Iter {e+1}' for e in selected_epochs])

# Lepše ozadje
fig.patch.set_facecolor('#f5f5f5')
ax.set_facecolor('#ffffff')

plt.tight_layout()
plt.show()
