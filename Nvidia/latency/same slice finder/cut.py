# Apri il file in modalità di lettura e leggi le righe
with open('index_raw.txt', 'r') as file:
    lines = file.readlines()

# Apri il file in modalità di scrittura e scrivi solo i primi 64 numeri di ogni riga
with open('index_raw_cut.txt', 'w') as file:
    for line in lines:
        numbers = line.split()  # Dividi la riga in numeri
        numbers = numbers[:128]  # Prendi solo i primi 64 numeri
        file.write(' '.join(numbers) + '\n')  # Scrivi i numeri nel file