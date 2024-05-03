# Nome del file di input e output
input_file = "addresses.txt"
output_file = "index.txt"

# Inizializza un dizionario per tenere traccia degli indici di start per ogni L2 block
l2_block_indices = {}

# Leggi le linee dal file di input e estrai le informazioni su L2 block, partizione e start_idx
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
        
        if l2_block in l2_block_indices:
            l2_block_indices[l2_block].append(start_idx)
        else:
            l2_block_indices[l2_block] = [start_idx]

# Ordina gli indici di start per ogni L2 block in ordine crescente
sorted_indices = []
for l2_block, indices in l2_block_indices.items():
    sorted_indices.append((l2_block, sorted(indices)))

# Ordina i risultati in base al numero di partizione e al numero di L2 block
sorted_indices = sorted(sorted_indices, key=lambda x: (int(x[0].split()[-1]), int(x[1][0])))

# Scrivi il risultato su un nuovo file di output
with open(output_file, "w") as outfile:
    for (l2_block, indices) in sorted_indices:
        outfile.write(f"L2 block {l2_block} ha gli indici: {', '.join(map(str, indices))}\n")