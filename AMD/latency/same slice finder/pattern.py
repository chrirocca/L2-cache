# Nome del file di input e output
input_file = "addresses.txt"
output_file = "addresses.txt"

# Apri il file di input in modalità lettura
with open(input_file, "r") as infile:
    # Leggi tutte le righe dal file di input
    lines = infile.readlines()

# Apri il file di output in modalità scrittura
with open(output_file, "w") as outfile:
    for i, line in enumerate(lines):
        # Scrivi la riga corrente nel file di output
        outfile.write(line)

        # Aggiungi una riga vuota ogni 16 righe
        if (i + 1) % 16 == 0:
            outfile.write("\n")

        # Aggiungi due righe vuote ogni 32 righe
        if (i + 1) % 32 == 0:
            outfile.write("\n\n")
