from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pdfplumber
import spacy
import streamlit as st
from docx import Document


SKILL_KEYWORDS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "sql",
    "html",
    "css",
    "react",
    "node.js",
    "django",
    "flask",
    "fastapi",
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "spacy",
    "nlp",
    "machine learning",
    "deep learning",
    "data analysis",
    "data visualization",
    "power bi",
    "tableau",
    "excel",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "linux",
    "api",
    "rest",
    "mongodb",
    "postgresql",
    "mysql",
    "spark",
}

SECTION_HEADERS = {
    "education": ["education", "academic background", "qualifications"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "projects",
    ],
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4})"
)
LINK_RE = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin\.com|github\.com)/\S+", re.I)


@dataclass
class CandidateProfile:
    name: str
    email: str
    phone: str
    skills: list[str]
    education: list[str]
    experience: list[str]
    links: list[str]
    source_file: str


@st.cache_resource(show_spinner=False)
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


def extract_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.read()

    if suffix == ".pdf":
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    if suffix == ".docx":
        document = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if suffix in {".md", ".txt"}:
        return data.decode("utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")


def clean_lines(text: str) -> list[str]:
    return [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]


def normalize_heading(line: str) -> str:
    return line.lower().strip(" #:")


def find_name(text: str, lines: list[str], nlp) -> str:
    doc = nlp("\n".join(lines[:12]))
    people = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    if people:
        return people[0]

    for line in lines[:8]:
        if EMAIL_RE.search(line) or PHONE_RE.search(line) or LINK_RE.search(line):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'.-]+", line)
        if 2 <= len(words) <= 4 and all(word[:1].isupper() for word in words[:2]):
            return " ".join(words)
    return ""


def find_skills(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    found = []
    for skill in SKILL_KEYWORDS:
        pattern = rf"(?<![\w+#.-]){re.escape(skill)}(?![\w+#.-])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(found)


def extract_section(lines: list[str], section_name: str, max_items: int = 8) -> list[str]:
    headers = SECTION_HEADERS[section_name]
    start_index = None
    for index, line in enumerate(lines):
        lowered = normalize_heading(line)
        if lowered in headers:
            start_index = index + 1
            break

    if start_index is None:
        return fallback_section(lines, section_name, max_items)

    collected: list[str] = []
    all_headers = {header for group in SECTION_HEADERS.values() for header in group}
    for line in lines[start_index:]:
        lowered = normalize_heading(line)
        if lowered in all_headers and collected:
            break
        if len(line) > 2:
            collected.append(line)
        if len(collected) >= max_items:
            break
    return collected


def fallback_section(lines: list[str], section_name: str, max_items: int) -> list[str]:
    if section_name == "education":
        education_terms = re.compile(
            r"\b(bachelor|master|phd|degree|diploma|university|college|school|b\.?s\.?|m\.?s\.?|mba)\b",
            re.I,
        )
        return [line for line in lines if education_terms.search(line)][:max_items]

    experience_terms = re.compile(
        r"\b(intern|developer|engineer|analyst|manager|consultant|experience|project|worked|built|led)\b",
        re.I,
    )
    return [line for line in lines if experience_terms.search(line)][:max_items]


def parse_resume(text: str, source_file: str) -> CandidateProfile:
    nlp = load_nlp()
    lines = clean_lines(text)
    email = next(iter(EMAIL_RE.findall(text)), "")
    phone = next(iter(PHONE_RE.findall(text)), "")

    return CandidateProfile(
        name=find_name(text, lines, nlp),
        email=email,
        phone=phone,
        skills=find_skills(text),
        education=extract_section(lines, "education"),
        experience=extract_section(lines, "experience"),
        links=sorted(set(LINK_RE.findall(text))),
        source_file=source_file,
    )


def profile_to_frame(profile: CandidateProfile) -> pd.DataFrame:
    data = asdict(profile)
    return pd.DataFrame(
        [
            {
                "field": key.replace("_", " ").title(),
                "value": ", ".join(value) if isinstance(value, list) else value,
            }
            for key, value in data.items()
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Resume Parser", layout="wide")

    st.title("Resume Parser")
    st.caption("Upload a resume and extract candidate details with regex, pandas, and SpaCy NLP.")

    uploaded_file = st.file_uploader(
        "Upload resume",
        type=["pdf", "md", "txt", "docx"],
        help="PDF, Markdown, TXT, and DOCX resumes are supported.",
    )

    if uploaded_file:
        try:
            raw_text = extract_text(uploaded_file)
            profile = parse_resume(raw_text, uploaded_file.name)
            frame = profile_to_frame(profile)

            left, right = st.columns([0.95, 1.05], gap="large")

            with left:
                st.subheader("Candidate Information")
                st.dataframe(frame, hide_index=True, use_container_width=True)

                export_json = json.dumps(asdict(profile), indent=2)
                st.download_button(
                    "Download JSON",
                    data=export_json,
                    file_name="candidate_profile.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.download_button(
                    "Download CSV",
                    data=frame.to_csv(index=False),
                    file_name="candidate_profile.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with right:
                st.subheader("Extracted Text")
                st.text_area("Preview", value=raw_text[:5000], height=520)

        except Exception as exc:
            st.error(f"Could not parse this file: {exc}")
    else:
        st.info("Upload a PDF, Markdown, TXT, or DOCX resume to begin.")


if __name__ == "__main__":
    main()
