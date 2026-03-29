import json
import re
import time
import unicodedata
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://admitere.unitbv.ro/"
LICENSE_FACULTIES_URL = "https://admitere.unitbv.ro/licenta-facultati.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
}

SECTION_KEYWORDS = [
    "programe de studii",
    "condiții de admitere",
    "conditii de admitere",
    "calendarul admiterii",
    "număr de locuri",
    "numar de locuri",
    "taxe",
    "acte necesare pentru înscriere",
    "acte necesare pentru inscriere",
    "informații suplimentare",
    "informatii suplimentare",
    "confirmarea locului",
    "contact",
    "cazare pentru candidații care susțin probe",
    "cazare pentru candidatii care sustin probe",
    "condiții de înscriere la al doilea program de licență",
    "conditii de inscriere la al doilea program de licenta",
    "înscrie-te!",
    "inscrie-te!",
    "tematica de admitere",
]

EXCLUDED_PAGE_PARTS = [
    "/informatii-masterat/",
    "/doctorat/",
    "/romani-de-pretutindeni/",
    "/rezultate/",
]

FOOTER_MARKERS = [
    "E-mail: admitere@unitbv.ro",
    "© 2026",
    "Politica de confidențialitate",
    "Detailed",
    "Personal $",
    "Business $",
    "Ultimate $",
]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ").strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def get_soup(url: str) -> BeautifulSoup:
    html = get_html(url)
    return BeautifulSoup(html, "lxml")


def is_valid_faculty_page(url: str) -> bool:
    if "/informatii-licenta/facultatea-de-" not in url:
        return False
    if not url.endswith(".html"):
        return False
    if any(part in url for part in EXCLUDED_PAGE_PARTS):
        return False

    excluded_subpages = [
        "/programe-de-studii.html",
        "/conditii-de-admitere.html",
        "/calendarul-admiterii.html",
        "/numar-de-locuri.html",
        "/taxe.html",
        "/acte-necesare",
        "/informatii-suplimentare.html",
        "/confirmarea-locului.html",
        "/contact.html",
        "/cazare",
        "/conditii-de-inscriere-la-al-doilea-program-de-licenta",
        "/tematica-de-admitere.html",
    ]
    if any(part in url for part in excluded_subpages):
        return False

    return True


def extract_faculty_pages() -> list[str]:
    soup = get_soup(LICENSE_FACULTIES_URL)
    urls = set()

    for a in soup.find_all("a", href=True):
        full_url = urljoin(BASE_URL, a["href"])
        if is_valid_faculty_page(full_url):
            urls.add(full_url)

    return sorted(urls)


def extract_faculty_name(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(["h1", "h2", "title", "li", "a"]):
        txt = clean_text(tag.get_text(" ", strip=True))
        if txt.startswith("Facultatea de"):
            txt = txt.replace(" - Admitere UNITBV", "").strip()
            return txt
    return ""


def extract_section_links(faculty_url: str) -> dict:
    soup = get_soup(faculty_url)
    faculty_name = extract_faculty_name(soup)

    sections = {}

    for a in soup.find_all("a", href=True):
        label = clean_text(a.get_text(" ", strip=True))
        if not label:
            continue

        label_norm = normalize_text(label)
        if label_norm not in SECTION_KEYWORDS:
            continue

        href = urljoin(faculty_url, a["href"])
        sections[label] = href

    return {
        "facultate": faculty_name,
        "faculty_url": faculty_url,
        "sections": sections
    }


def get_candidate_containers(soup: BeautifulSoup):
    candidates = []
    tags = soup.find_all(["main", "article", "section", "div"])

    for tag in tags:
        text = tag.get_text(" ", strip=True)
        text_len = len(text)

        if text_len < 120:
            continue

        p_count = len(tag.find_all("p"))
        li_count = len(tag.find_all("li"))
        h_count = len(tag.find_all(["h1", "h2", "h3", "h4"]))
        a_count = len(tag.find_all("a"))

        score = text_len + (p_count * 80) + (li_count * 60) + (h_count * 40) - (a_count * 15)
        candidates.append((score, tag))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [tag for score, tag in candidates]


def extract_structured_text_from_container(container) -> str:
    pieces = []

    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li"], recursive=True):
        text = clean_text(el.get_text(" ", strip=True))
        if not text:
            continue

        if el.name in ["h1", "h2", "h3", "h4"]:
            pieces.append(text)
        elif el.name == "li":
            pieces.append(f"• {text}")
        else:
            pieces.append(text)

    result = "\n".join(pieces)
    result = re.sub(r"\n{2,}", "\n", result).strip()
    return result


def looks_like_noise(text: str) -> bool:
    if not text:
        return True

    noise_patterns = [
        "Facultatea de Design de produs și mediu",
        "Facultatea de Inginerie electrică și știința calculatoarelor",
        "Facultatea de Design de mobilier și inginerie a lemnului",
        "Rezultate Medii admitere",
        "Politica de confidențialitate",
        "Toggle navigation",
    ]

    hit_count = sum(1 for pat in noise_patterns if pat in text)
    return hit_count >= 2


def remove_footer_noise(text: str) -> str:
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def remove_faculty_prefix(text: str, faculty_name: str) -> str:
    if not text:
        return text

    patterns = [
        rf"^{re.escape(faculty_name)}\s*\|\s*",
        rf"^-?\s*{re.escape(faculty_name)}\s*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def extract_programs_only(text: str) -> str:
    start_markers = [
        "Învățământ cu frecvență",
        "Învățământ la distanță",
        "Învățământ cu frecvență redusă",
    ]

    start_idx = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx
            break

    if start_idx != -1:
        text = text[start_idx:]

    end_markers = [
        "Vezi mai multe:",
        "E-mail:",
        "© 2026",
        "Politica de confidențialitate",
    ]

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    return text.strip()


def clean_contact(text: str) -> str:
    start_markers = ["Adresa facultății:", "E-mail:", "Telefon:"]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]
            break

    end_markers = ["© 2026", "Politica de confidențialitate"]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    return text.strip()


def clean_confirmare_loc(text: str) -> str:
    start_markers = [
        "CALENDAR CONFIRMARE LOCURI ADMITERE",
        "Informații pentru candidații admiși",
    ]

    start_idx = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx
            break

    if start_idx != -1:
        text = text[start_idx:]

    end_markers = [
        "Ştiri și evenimente",
        "Știri și evenimente",
        "© 2026",
        "Politica de confidențialitate",
    ]

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    return text.strip()


def clean_page_content(text: str, tip_info: str, faculty_name: str) -> str:
    text = clean_text(text)
    text = remove_footer_noise(text)

    if tip_info == "programe_studiu":
        text = extract_programs_only(text)
    elif tip_info == "confirmare_loc":
        text = clean_confirmare_loc(text)
    elif tip_info == "contact":
        text = clean_contact(text)

    text = remove_faculty_prefix(text, faculty_name)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_tip_info(section_name: str) -> str:
    section_norm = normalize_text(section_name)

    if "programe de studii" in section_norm:
        return "programe_studiu"
    if "conditii de admitere" in section_norm or "condiții de admitere" in section_norm:
        return "conditii_admitere"
    if "calendarul admiterii" in section_norm:
        return "calendar_admitere"
    if "numar de locuri" in section_norm or "număr de locuri" in section_norm:
        return "numar_locuri"
    if "taxe" in section_norm:
        return "taxe"
    if "acte necesare" in section_norm:
        return "acte_inscriere"
    if "informatii suplimentare" in section_norm or "informații suplimentare" in section_norm:
        return "informatii_suplimentare"
    if "confirmarea locului" in section_norm:
        return "confirmare_loc"
    if "contact" in section_norm:
        return "contact"
    if "cazare" in section_norm:
        return "cazare_probe"
    if "al doilea program de licenta" in section_norm or "al doilea program de licență" in section_norm:
        return "al_doilea_program"
    if "inscrie-te" in section_norm or "înscrie-te" in section_norm:
        return "inscriere_online"
    if "tematica de admitere" in section_norm:
        return "tematica"
    return "pagina_admitere"


def extract_main_content_from_page(url: str, section_name: str, tip_info: str, faculty_name: str) -> str:
    soup = get_soup(url)

    for tag in soup(["script", "style", "noscript", "footer", "header", "nav", "aside"]):
        tag.decompose()

    candidates = get_candidate_containers(soup)

    best_text = ""
    for container in candidates[:10]:
        text = extract_structured_text_from_container(container)
        if not text:
            continue
        if looks_like_noise(text) and tip_info not in {"contact", "confirmare_loc", "programe_studiu"}:
            continue
        best_text = text
        break

    if not best_text:
        best_text = clean_text(soup.get_text("\n", strip=True))

    best_text = clean_page_content(best_text, tip_info, faculty_name)
    return best_text


def scrape_all():
    faculty_pages = extract_faculty_pages()
    print(f"Am găsit {len(faculty_pages)} pagini de facultate.")

    structure = []
    pages_content = []

    for faculty_url in faculty_pages:
        try:
            faculty_data = extract_section_links(faculty_url)
            structure.append(faculty_data)

            print(f"[FACULTATE] {faculty_data['facultate']} | {len(faculty_data['sections'])} secțiuni")

            for section_name, section_url in faculty_data["sections"].items():
                try:
                    tip_info = infer_tip_info(section_name)
                    content = extract_main_content_from_page(
                        section_url,
                        section_name,
                        tip_info,
                        faculty_data["facultate"]
                    )

                    page_record = {
                        "facultate": faculty_data["facultate"],
                        "faculty_url": faculty_url,
                        "section_name": section_name,
                        "tip_info": tip_info,
                        "section_url": section_url,
                        "content": content
                    }
                    pages_content.append(page_record)
                    print(f"   [OK] {section_name}")
                    time.sleep(0.4)

                except Exception as e:
                    print(f"   [EROARE PAGINĂ] {section_name} -> {e}")

            time.sleep(0.4)

        except Exception as e:
            print(f"[EROARE FACULTATE] {faculty_url} -> {e}")

    return structure, pages_content


def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    structure, pages_content = scrape_all()

    save_json(structure, "scraped_content/unitbv_structure.json")
    save_json(pages_content, "scraped_content/unitbv_pages.json")

    print("\nGata.")
    print("- scraped_content/unitbv_structure.json")
    print("- scraped_content/unitbv_pages.json")