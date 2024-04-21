import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import requests

def read_file(file_path):
    try:
        if file_path.endswith('.pdf'):
            pdf_file_obj = open(file_path, 'rb')
            pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
            num_pages = pdf_reader.numPages
            text = ''
            for page in range(num_pages):
                page_obj = pdf_reader.getPage(page)
                text += page_obj.extractText()
            pdf_file_obj.close()
            return text

        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            text = ''
            for para in doc.paragraphs:
                text += para.text
            return text

        elif file_path.startswith('http://') or file_path.startswith('https://'):
            response = requests.get(file_path)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            return text

        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text

    except Exception as e:
        print(f"Error reading file: {e}")
        return None