# Nome del file di input e output
input_file = "index.txt"
output_file = "index.txt"

# Leggere le righe dal file di input e salvarle in una lista
with open(input_file, "r") as infile:
    lines = infile.readlines()

# Definire una funzione per estrarre il numero dalla riga
def extract_number(line):
    # Divide la riga in base alle virgole
    parts = line.split(",")
    if len(parts) >= 2:
        indices_info = parts[0].strip()  # Ottieni l'informazione sugli indici
        
        # Estrai il primo numero dopo ":" e prima di ","
        number = int(indices_info.split(":")[1].split(",")[0])
        
        return number
    return 0  # Restituisci un valore di default

# Ordina le righe in base al numero estratto dalla riga
sorted_lines = sorted(lines, key=extract_number)

# Scrivere le righe ordinate nel file di output
with open(output_file, "w") as outfile:
    outfile.writelines(sorted_lines)
