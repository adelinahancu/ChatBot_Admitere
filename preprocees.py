import pdfplumber
import unicodedata
import re


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r'\s+', ' ', text)
    return text


def clean_pdf_structured(pdf_path, output_path):
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_path, "w", encoding="utf-8") as f:
            current_facultate = ""
            current_domeniu = ""

            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue

                for row in table:
                    row = [str(c).replace('\n', ' ').strip() if c else "" for c in row]

                    if not row or "Nr. ordine" in row[0] or "TOTAL" in row:
                        continue

                    if len(row) < 8:
                        continue

                    if row[1]:
                        current_facultate = row[1]
                    if row[2]:
                        current_domeniu = row[2]

                    program = row[3]

                    if program and any(n.strip() for n in row[4:]):
                        program_norm = normalize_text(program)
                        facultate_norm = normalize_text(current_facultate)
                        domeniu_norm = normalize_text(current_domeniu)

                        # Boost textual pentru BM25 / retrieval
                        line = (
                            f"FACULTATE: {current_facultate} | "
                            f"DOMENIU: {current_domeniu} | "
                            f"PROGRAM: {program} | "
                            f"NUME_PROGRAM: {program} | "
                            f"ALIAS_PROGRAM: {program_norm} {program_norm} {program_norm} | "
                            f"FACULTATE_NORMALIZATA: {facultate_norm} | "
                            f"DOMENIU_NORMALIZAT: {domeniu_norm} | "
                            f"MEDII: {' | '.join(row[4:])}\n"
                        )

                        f.write(line)


if __name__ == "__main__":
    clean_pdf_structured("source_docs/Medii_admitere_2024.pdf", "source_docs/date_curate2.md")