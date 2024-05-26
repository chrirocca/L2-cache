import os
import re
import matplotlib.pyplot as plt

def cm2inch(value):
    return value/2.54

# Definisci le etichette per gli assi x e y
x_label = "# SMs"
y_label = "Bandwidth [GB/s]"

# Crea le liste vuote per i dati da plot
x_values = list(range(1, 81, 1))  # Da 1 a 32
y_values = []

# Per ogni numero di warp_max da 0 a 31
for warp_max in range(1,81):
    log_file_name = f"slices8_sms{warp_max}.log"

    # Leggi l'ultimo numero dalla riga 11 del file di log
    with open(log_file_name, "r") as file:
        lines = file.readlines()
        last_line = lines[10].strip()  # Riga 11 è l'indice 10 nella lista
        last_value = last_line.split()[-1]  # Estrai l'ultimo valore dalla riga

        if "KB/s" in last_value:
            last_number = float(re.findall(r'\d+\.\d+', last_value)[0]) / 10**6
        elif "MB/s" in last_value:
            last_number = float(re.findall(r'\d+\.\d+', last_value)[0]) / 10**3
        elif "GB/s" in last_value:
            last_number = float(re.findall(r'\d+\.\d+', last_line)[-1])

    # Aggiungi il valore a y_values
    y_values.append(last_number)


plt.figure(figsize=(cm2inch(16), cm2inch(8)))  # Esempio: 12 pollici di larghezza per 6 pollici di altezza

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.bar(x_values, y_values, color='#1f77b4')

# Crea il grafico a barre
plt.bar(x_values, y_values)
plt.xticks(x_values, rotation=90) # Rotate x-axis ticks by 90 degrees
plt.xlabel(x_label, fontsize=14)
plt.ylabel(y_label, fontsize=14)
plt.xticks(fontsize=9) # Increase font size of x-axis labels
plt.yticks(fontsize=9) # Increase font size of y-axis labels
plt.yticks(range(0, 750, 150)) # Set ticks on y-axis every 10
plt.xticks(range(1, 81, 10)) # Set ticks on y-axis every 10
plt.ylim(0, 750)  # Imposta il limite superiore dell'asse y

plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', markersize=3)

# Salva il grafico in un file
plt.savefig("throughput_increased.png", dpi=450,bbox_inches='tight')