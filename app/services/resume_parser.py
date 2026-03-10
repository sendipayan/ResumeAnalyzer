import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import fitz
from typing import List


class CVPipeline:

  # =========================
  # CONFIG
  # =========================

  FEATURE_HEADERS = {
                      "summary": ["summary", "professional summary", "career summary", "profile"],
                      "experience": [
                                    "experience",
                                    "work experience",
                                    "professional experience",
                                    "employment history",
                                     "work history"
                                    ],
                      "skills": ["skills", "technical skills", "core skills"],
                      "education": ["education", "academic background",        "education and training"],
                      "projects": ["projects", "personal projects", "proof of work"],
                      "achievements": [
                                      "achievements",
                                      "awards",
                                      "honors",
                                      "accomplishments",
                                      "certifications",
                                      "recognition"
                                      ]
  }
  MONTHS_PATTERN = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"

  # =========================
  # TEXT EXTRACTION
  # =========================

  def pdf_from_url(self, url: str, timeout: int = 20, max_pdf_bytes: int = 5 * 1024 * 1024) -> str:
    import requests
    from io import BytesIO
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=False) as response:
      response.raise_for_status()

      pdf_stream = BytesIO()
      for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
          continue
        pdf_stream.write(chunk)
        if pdf_stream.tell() > max_pdf_bytes:
          raise ValueError(f"PDF exceeds maximum allowed size ({max_pdf_bytes} bytes)")

    pdf_stream.seek(0)
    pdf_bytes = pdf_stream.read()
    if not pdf_bytes:
      raise ValueError("Downloaded file is empty")

    # Accept providers that return octet-stream by verifying actual PDF magic bytes.
    if b"%PDF-" not in pdf_bytes[:1024]:
      raise ValueError("URL did not return a valid PDF document")

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
      return "\n".join(page.get_text("text") for page in doc)


  def pdf_to_text(self, pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text

  # =========================
  # CLEANING
  # =========================

  def clean_text(self, text: str) -> str:
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

  def clean_line(self, text: str) -> str:
    if not isinstance(text, str):
      return ""
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[\u2022\u25cf\u25cb\u25aa\uf0b7\xb7\*\•]", " , ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")

  # =========================
  # CONTACT EXTRACTION
  # =========================

  def extract_email(self, text: str):
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else ""

  def extract_phone(self, text: str):
    match = re.search(r"(\+?\d{1,3}[\s-]?)?\(?\d{3,5}\)?[\s-]?\d{3,5}[\s-]?\d{3,5}", text)
    return match.group(0) if match else ""

  def extract_name(self, text: str):
    lines = text.split("\n")[:10]

    for line in lines:
      line = line.strip()

      if not line:
        continue

      if self.extract_email(line) or self.extract_phone(line):
        continue

# Reject obvious job titles
      blacklist = ["engineer", "developer", "consultant",
                   "manager", "management", "specialist",
                   "analyst", "director", "technology"]

      if any(word in line.lower() for word in blacklist):
        continue

      words = line.split()

# Likely name: 2–4 words, alphabetic, not all caps tech phrase
      if 1 < len(words) <= 4 and all(w.isalpha() for w in words):
        return line.title()

    return ""
# =========================
# SKILL LIST
# =========================





  def parse_skill_section(self, text):
    all_skills = []

    # Detect category headers
    category_headers = re.findall(r'\b([\w\s/+-]{2,30}):\s*[a-zA-Z]', text)

    if len(category_headers) > 0:
      parts = re.split(r'\b([\w\s/+-]{2,30}):', text)

      for i in range(1, len(parts), 2):
        skills_text = parts[i + 1]

        skills = [
                  s.strip()
                  for s in skills_text.split(",")
                  if s.strip()
        ]

        all_skills.extend(skills)

    else:
      all_skills = [
                    s.strip()
                    for s in text.split(",")
                    if s.strip()
      ]

    return all_skills




# =========================
# SECTION DETECTION
# =========================

  def is_header_line(self, line: str):
    line = line.strip().lower()
    if not line:
      return False
    if len(line) > 40:
      return False
    if line.endswith("."):
      return False
    if len(line.split()) > 5:
      return False
    return True

  def match_header(self, line: str):
    line = line.strip().lower()
    for section, headers in self.FEATURE_HEADERS.items():
      for header in headers:
        if line == header or line.startswith(header + ":"):
          return section
    return None

  def split_sections(self, text: str):
    lines = text.split("\n")
    sections = {k: "" for k in self.FEATURE_HEADERS}
    current = None

    for raw_line in lines:
      line = raw_line.strip()

      if self.is_header_line(line):
        matched = self.match_header(line)
        if matched:
          current = matched
          continue

      if current:
        if current == "skills":
          sections[current] += ", " + raw_line

        else:
          sections[current] += " " + raw_line


    return {k: v.strip() for k, v in sections.items()}

# =========================
# EXPERIENCE ENRICHMENT
# =========================

  def calculate_experience_years(self, text: str):
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)

    years = [int(y) for y in years]

    if len(years) >= 2:
      return max(years) - min(years)

    return 0.0


# =========================
# EDUCATION ENRICHMENT
# =========================


  def certificates(self, text: str) -> List[str]:
    if not text:
        return []

    certs = re.findall(
        r"\b([A-Z][A-Za-z0-9+&\-/ ]{2,60}?\s(?:certified|certification|certificate|license|licence|credential)s?)\b",
        text,
        re.I
    )

    # Clean + deduplicate while preserving order
    seen = set()
    result = []
    for c in certs:
        clean = c.strip()
        if clean.lower() not in seen:
            seen.add(clean.lower())
            result.append(clean)

    return result

  def enrich_education(self, text: str):
    if not text:
      return ""

    


    institutions = re.findall(
                            r"((?:\b\w+\b\s+){1,3}(university|college|institute|school|polytechnic))",
                            text,
                            re.I
                            )

    institutions = ", ".join({i[0].strip() for i in institutions}) or "unknown"

    

    return institutions
  def split_projects(project_text):
    # Split where a new project title likely starts
    projects = re.split(r'\n(?=[A-Z][^\n]+\n)', project_text.strip())

    return [p.strip() for p in projects if p.strip()]

# =========================
# MAIN RUN
# =========================

  def run(self, pdf_url: str, download_timeout: int = 20, max_pdf_bytes: int = 5 * 1024 * 1024):




    raw_text = self.pdf_from_url(
      pdf_url,
      timeout=download_timeout,
      max_pdf_bytes=max_pdf_bytes
    )
    cleaned = raw_text

    sections = self.split_sections(cleaned)


    row = {


              "resume_text": raw_text,
              "summary": self.clean_line(sections.get("summary", "")),
              "experience": self.clean_line(sections.get("experience", "")),
              "skills": self.clean_line(sections.get("skills", "")),
              "education": self.clean_line(sections.get("education", "")),
              "projects": sections.get("projects", ""),
              "achievements": self.clean_line(sections.get("achievements",""))
            }


    row["skills_list"] = self.parse_skill_section(sections["skills"])
    row["experience_years"] = self.calculate_experience_years(row["experience"])
    row["education_enriched"] = self.enrich_education(row["education"])
    row["certificates"]=self.certificates(row["education"]+" "+row["summary"])



    return row



