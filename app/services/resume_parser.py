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
                      "summary": ["summary", "professional summary", "career summary", "profile", "profile summary"],
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
  

  def parse_contact_info(self,text):

    contact = {
        "name": None,
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None
    }


    # EMAIL (robust to PDF line breaks and spacing around @ / .)
    email_text = text.replace("\u200b", "")
    email_text = re.sub(r"\s*@\s*", "@", email_text)
    email_text = re.sub(r"\s*\.\s*", ".", email_text)
    email_pattern = r'\b(?![A-Za-z0-9._%+-]*\.\.)[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b'
    email_match = re.search(email_pattern, email_text)
    if email_match:
        contact["email"] = email_match.group()

    # PHONE
    phone_pattern = r'(\+?\d{1,3}[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        contact["phone"] = phone_match.group()

    # LINKEDIN
    linkedin_pattern = r'(https?:\/\/)?(www\.)?linkedin\.com\/[A-Za-z0-9_\-\/]+'
    linkedin_match = re.search(linkedin_pattern, text)
    if linkedin_match:
        linkedin = linkedin_match.group()
        if not linkedin.startswith("http"):
            linkedin = "https://" + linkedin
        contact["linkedin"] = linkedin

    # GITHUB
    github_pattern = r'(https?:\/\/)?(www\.)?github\.com\/[A-Za-z0-9_\-]+'
    github_match = re.search(github_pattern, text)
    if github_match:
        github = github_match.group()
        if not github.startswith("http"):
            github = "https://" + github
        contact["github"] = github

    # NAME (reuse extract_name heuristic, then fallback for ALL CAPS names)
    lines = text.split("\n")

    name = self.extract_name(text)
    if not name:
        for line in lines[:10]:
            raw = line.strip()
            if re.match(r'^[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+){1,3}$', raw):
                name = raw.title()
                break
    contact["name"] = name or None

    return contact
  
  def text_readability_score(self,text):
    if len(text.strip()) < 500:
        return 20
    return 100


# -------- TABLE DETECTION --------
  def detect_tables(self,text):

    table_patterns = [
        r"\|.+\|",      # markdown style tables
        r"\t",          # tab columns
        r"\s{5,}"       # large spaces between columns
    ]

    for pattern in table_patterns:
        if re.search(pattern, text):
            return True

    return False


# -------- COLUMN DETECTION --------
  def column_score(self,text):

    lines = text.split("\n")

    short_lines = sum(1 for l in lines if len(l.strip()) < 20 and len(l.strip()) > 0)
    total = len(lines)

    if total == 0:
        return 50

    ratio = short_lines / total

    if ratio > 0.35:
        return 40

    return 100


# -------- SPECIAL CHARACTER DETECTION --------
  def special_char_score(self,text):

    special_chars = re.findall(r'[^\x00-\x7F]+', text)

    if len(special_chars) > 10:
        return 40

    return 100


# -------- BULLET STRUCTURE --------
  def bullet_score(self,text):

    bullets = re.findall(r'•|-|\*', text)

    if len(bullets) > 5:
        return 100

    if len(bullets) > 0:
        return 80

    return 60


# -------- SPACING CONSISTENCY --------
  def spacing_score(self,text):

    double_spaces = text.count("  ")

    if double_spaces > 50:
        return 60

    return 100


# -------- FORMATTING SCORE --------
  def formatting_score(self,text):

    readability = self.text_readability_score(text)
    table = 0 if self.detect_tables(text) else 100
    column = self.column_score(text)
    special = self.special_char_score(text)
    bullet = self.bullet_score(text)
    spacing = self.spacing_score(text)

    score = (
        0.20 * readability +
        0.20 * table +
        0.20 * column +
        0.15 * special +
        0.15 * bullet +
        0.10 * spacing
    )

    return round(score, 2)
  
  def ATS_score(self,pdf_url: str, download_timeout: int = 20, max_pdf_bytes: int = 5 * 1024 * 1024):

    text = self.pdf_from_url(
      pdf_url,
      timeout=download_timeout,
      max_pdf_bytes=max_pdf_bytes
    )
    
    if not text.strip():
      return {
        "ATS_score": 0,
        "section_score": 0,
        "contact_score": 0,
        "formating_score":0,
        "issues":["No text detected in resume"]
      }
    
    section = self.split_sections(text)
    contact=self.parse_contact_info(text)
    formatting_score=self.formatting_score(text)
    
    issues=[]
    
    section_score=0
    if section["summary"] :
      section_score+=15
    else :
      issues.append("Summary section not detected")
    if section ["skills"] :
      section_score+=25
    else :
      issues.append("Skills section not detected")
    if section ["education"] :
      section_score+=15
    else :
      issues.append("Education section not detected")
    if section ["projects"] or section["experience"] :
      section_score+=30
    else :
      issues.append("Projects or Experience section not detected")
    if section ["achievements"] :
      section_score+=15
    else :
      issues.append("Achievements section not detected")

    contact_score = 0

    if contact["name"]:
      contact_score += 20
    else :
      issues.append("Name not found")
    if contact["email"]:
      contact_score += 30
    else :
      issues.append("Email not found")
    if contact["phone"]:
      contact_score += 30
    else :
      issues.append("Phone No. not found")
    if contact["linkedin"] or contact["github"]:
      contact_score += 20
    else :
      issues.append("Linkedin or github not found")

    if self.detect_tables(text):
        issues.append("Tables detected (ATS may struggle)")

    if self.column_score(text) < 100:
        issues.append("Possible multi-column layout detected")

    if self.special_char_score(text) < 100:
        issues.append("Special characters/icons detected")

    if self.bullet_score(text) < 80:
        issues.append("Weak bullet structure")

    if self.spacing_score(text) < 100:
        issues.append("Inconsistent spacing")

    if len(text.strip()) < 500:
        issues.append("Low extractable text (possible image PDF)")

    ATS_Score = 0.40 * section_score + 0.25 * contact_score + 0.35 * formatting_score

    return {
        "text": text,
        "ATS_score": ATS_Score,
        "section_score": section_score,
        "contact_score": contact_score,
        "formating_score":formatting_score,
        "issues": issues
    }


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
    contact=self.parse_contact_info(cleaned)

    row = {
              "contact": contact,
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



