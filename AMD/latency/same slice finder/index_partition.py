# Nome del file di input e output
input_file = "addresses.txt"
output_file = "index.txt"

# Creare un dizionario per tenere traccia degli indici per ogni valore di L2 block e partizione
l2_block_indices = {}

# Leggere i dati dal file di input
with open(input_file, "r") as infile:
    lines = infile.readlines()

for line in lines:
    parts = line.split(",")
    if len(parts) >= 3:
        l2_block_info = parts[0].strip()  # Ottieni l'informazione su L2 block
        partition_info = parts[1].strip()  # Ottieni l'informazione sulla partizione
        start_idx_info = parts[2].strip()  # Ottieni l'informazione su start_idx
        
        # Estrai il numero di L2 block, partizione e start_idx
        l2_block = l2_block_info.split()[-1]
        partition_number = int(partition_info.split()[-1])
        start_idx = int(start_idx_info.split()[-1])
        
        # Crea una chiave unica per la combinazione di partizione e L2 block
        key = (partition_number, l2_block)
        
        if key in l2_block_indices:
            l2_block_indices[key].append(start_idx)
        else:
            l2_block_indices[key] = [start_idx]

# Ordina il dizionario in base al valore del primo indice di ogni L2 block e partizione
sorted_indices = sorted(l2_block_indices.items(), key=lambda x: (int(x[0][0]), int(x[0][1])))

# Scrivi il risultato su un file di output
with open(output_file, "w") as outfile:
    for (partition_number, l2_block), indices in sorted_indices:
        outfile.write(f"Partition {partition_number}, L2 block {l2_block} ha gli indici: {', '.join(map(str, sorted(indices)))}\n")
