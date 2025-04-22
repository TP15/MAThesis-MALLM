import os
import re

def remove_links_from_mal_files():
    # Regex zum Erkennen von URLs
    url_pattern = re.compile(r'https?://\S+')

    for filename in os.listdir():
        if filename.endswith('.mal'):
            with open(filename, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                stripped_line = line.strip()
                # Zeile löschen, wenn sie mit "user info" oder "developer info" beginnt und einen Link enthält
                if (stripped_line.startswith("user info") or stripped_line.startswith("developer info")) and url_pattern.search(line):
                    continue
                else:
                    # Link entfernen, aber Zeile behalten
                    cleaned_line = url_pattern.sub('', line)
                    new_lines.append(cleaned_line)

            # Datei überschreiben mit den bereinigten Zeilen
            with open(filename, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)

if __name__ == "__main__":
    remove_links_from_mal_files()

    #remoeved all the copyrights and added the basic language to all the files
