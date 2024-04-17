import re

# Nazwa pliku wejściowego
input_file = 'segmentation.csv'
# Nazwa pliku wyjściowego
output_file = 'segmentation_modified.csv'

# Funkcja do usuwania cudzysłowów z całej linii
def remove_quotes(line):
    return line.replace('"', '').replace("'", '')

# Otwieramy plik wejściowy do odczytu
with open(input_file, 'r') as file_in:
    # Otwieramy plik wyjściowy do zapisu
    with open(output_file, 'w') as file_out:
        # Przetwarzamy każdą linijkę pliku wejściowego
        for line in file_in:
            # Usuwamy cudzysłowy z całej linii
            line = remove_quotes(line)

            # Szukamy wystąpienia znaku "{" w linijce
            start_index = line.find('{')
            if start_index != -1:
                # Znaleziono znak "{", znajdujemy znak "}" dalej po "{" i zastępujemy średniki
                end_index = line.find('}', start_index)
                if end_index != -1:
                    # Fragment pomiędzy "{" a "}"
                    block_to_process = line[start_index:end_index + 1]
                    # Zamieniamy wystąpienia "UNKN" na "null"
                    processed_block = block_to_process.replace('UNKN', 'null')
                    # Zamieniamy wszystkie wystąpienia ";" na ","
                    processed_block = re.sub(';', ',', processed_block)
                    
                    # Dodajemy cudzysłowy przed przecinkiem oraz za dwukropkiem
                    processed_block = processed_block.replace(',', '", "').replace(':', '": "')

                    if end_index-start_index>2:
                        processed_block = processed_block.replace('{', '{ "').replace('}', '" }')

                    # Zamieniamy oryginalny fragment w linijce na przetworzony
                    line = line[:start_index] + processed_block + line[end_index + 1:]
            
            # Zapisujemy przetworzoną linijkę do pliku wyjściowego
            file_out.write(line)

print(f'Przetworzono plik {input_file}. Wynik zapisano do {output_file}.')
