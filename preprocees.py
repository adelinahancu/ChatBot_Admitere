import pdfplumber

def clean_pdf_structured(pdf_path, output_path):
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_path, "w", encoding="utf-8") as f:
            current_facultate = ""
            current_domeniu = ""
            
            for page in pdf.pages:
                table = page.extract_table()
                if not table: continue
                
                for row in table:
                    # Curățăm celulele de newline-uri
                    row = [str(c).replace('\n', ' ').strip() if c else "" for c in row]
                    
                    # Verificăm dacă rândul are date (evităm headerele sau rândurile goale)
                    if not row or "Nr. ordine" in row[0] or "TOTAL" in row:
                        continue

                    # Logica de "umplere" a celulelor goale (merge cells)
                    # În tabelul tău, Facultatea e pe coloana 1, Domeniul pe 2
                    if row[1]: current_facultate = row[1]
                    if row[2]: current_domeniu = row[2]
                    
                    # Programul este pe coloana 3, Notele încep de la 4
                    program = row[3]
                    note = " | ".join(row[4:])
                    
                    if program and any(n.strip() for n in row[4:]):
                        line = f"{current_facultate} | {current_domeniu} | {program} | {note}\n"
                        f.write(line)

clean_pdf_structured("source_docs/Medii_admitere_2024.pdf", "source_docs/date_curate.md")