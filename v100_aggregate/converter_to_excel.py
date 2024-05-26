import openpyxl

# Leggi il file di testo
file_txt = 'result.txt'
with open(file_txt, 'r', encoding='utf-8') as txt_file:
    lines = txt_file.readlines()

# Rimuovi tutte le virgole dalle linee del testo
lines = [line.replace(',', '') for line in lines]

# Crea un nuovo file Excel
workbook = openpyxl.Workbook()
sheet = workbook.active

# Scrivi ogni parola in una colonna diversa
for row_idx, line in enumerate(lines, start=1):
    words = line.split()
    for col_idx, word in enumerate(words, start=1):
        sheet.cell(row=row_idx, column=col_idx, value=word)

# Salva il file Excel
file_xlsx = 'result.xlsx'
workbook.save(file_xlsx)

