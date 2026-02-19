import requests
from bs4 import BeautifulSoup
import json

def extract_faq():
    url="https://admitere.unitbv.ro/intrebari-frecvente.html"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8'

    if response.status_code !=200:
        print("Eroare la accesarea site-ului!")
        return
    
    soup = BeautifulSoup(response.text,'html.parser')
    faq_data = []

    groups = soup.find_all('div',class_='accordion-group')

    for group in groups:
        question_tag = group.find('a',class_='accordion-toggle')
        response_tag = group.find('div', class_='accordion-inner')

        if question_tag and response_tag:
            q=question_tag.get_text(strip=True)
            a=response_tag.get_text(strip=True)
            faq_data.append({'question':q , 'answer':a})

    with open("date_admitere.json","w", encoding='utf-8') as f:
        json.dump(faq_data,f,ensure_ascii=False,indent=4)

    print(f"All done!Saved {len(faq_data)} paires of question-answer in 'date_admitere.json'.")

if __name__ == "__main__":
    extract_faq()
