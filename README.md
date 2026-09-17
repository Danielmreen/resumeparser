# Resume Parser with SpaCy NLP

A small Streamlit app that accepts resume uploads and extracts structured candidate information:

- name
- email
- phone
- skills
- education
- work experience signals
- links
- extracted text preview

## Supported uploads

- PDF
- Markdown
- TXT
- DOCX

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The app also works without the SpaCy model, but name extraction is better with `en_core_web_sm`.

## Run

```powershell
streamlit run app.py
```

## Notes

This is a learning-project parser, so it uses a practical combination of regex, dictionaries, and NLP entity detection. Production resume parsers usually combine larger labelled datasets, layout-aware PDF parsing, custom NER training, and validation workflows.
