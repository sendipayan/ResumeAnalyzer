from rapidfuzz import process, fuzz

import re
import pandas as pd
import numpy as np


def cosine_similarity(a, b):
  a_arr = np.array(a, dtype=np.float32)
  b_arr = np.array(b, dtype=np.float32)
  if a_arr.ndim == 1:
    a_arr = a_arr.reshape(1, -1)
  if b_arr.ndim == 1:
    b_arr = b_arr.reshape(1, -1)
  a_norm = np.linalg.norm(a_arr, axis=1, keepdims=True)
  b_norm = np.linalg.norm(b_arr, axis=1, keepdims=True)
  a_norm = np.where(a_norm == 0, 1e-12, a_norm)
  b_norm = np.where(b_norm == 0, 1e-12, b_norm)
  return (a_arr @ b_arr.T) / (a_norm * b_norm.T)



class RolePredicter:



  def normalize(self,skill):
    skill = skill.lower().strip()
    skill = skill.replace(".js", "js")
    skill = skill.replace(" ", "")
    skill = re.sub(r"[^\w#+]", "", skill)
    return skill

  

  def normalize_text(self, skill):

    skill = skill.lower().strip()

    # normalize common tech variations
    replacements = {
        ".js": "js",
        "node.js": "nodejs",
        "react.js": "reactjs",
        "next.js": "nextjs",
        "vue.js": "vuejs",
        "tailwindcss": "tailwind css",
        "machine-learning": "machine learning",
        "deep-learning": "deep learning"
    }

    for k, v in replacements.items():
        skill = skill.replace(k, v)

    # remove version numbers (html5 → html, css3 → css)
    skill = re.sub(r'(html|css|es)\d+', r'\1', skill)

    # normalize separators
    skill = re.sub(r"[-_/]", " ", skill)

    # keep alphanumeric + # +
    skill = re.sub(r"[^\w#+]", " ", skill)

    # collapse spaces
    skill = re.sub(r"\s+", " ", skill).strip()

    return skill


  def apply_alias(self,skill):
    return self.ALIAS_MAP.get(skill, skill)

  def build_role_index(self,role_skills):
    return {
            self.normalize(s): s for s in role_skills
    }

  def match_skill(self,resume_skill, role_skills, threshold=85):
    resume_skill = self.normalize(resume_skill)


    # Build normalized map (normalized → original)
    normalized_map = {
                      self.normalize(s): s for s in role_skills
    }

    normalized_role_skills = list(normalized_map.keys())

    # Special rule for short skills (avoid fuzzy chaos)
    if len(resume_skill) <= 2:
      if resume_skill in normalized_role_skills:
        return normalized_map[resume_skill], 100
      return None, 0

    result = process.extractOne(
                                resume_skill,
                                normalized_role_skills,
                                scorer=fuzz.token_sort_ratio
    )

    if not result:
      return None, 0

    match, score, _ = result

    if score >= threshold:
      return normalized_map[match], score
    return None, score




  def compare_skill_lists(self, role_skills, threshold=85):
    role_index = self.build_role_index(role_skills)

    matched_role_skills = set()
    matched_pairs = []
    missed_skills=[]

    for r_skill in self.resume_skills:
      match, score = self.match_skill(r_skill, role_skills, threshold)

      if match:
        normalized_match = self.normalize(match)
        if normalized_match in matched_role_skills:
          continue
        matched_role_skills.add(normalized_match)
        matched_pairs.append(r_skill)

      if not match:
        missed_skills.append(r_skill)



    total_role_skills = len(role_skills)

    coverage_score = (len(matched_role_skills) / total_role_skills
                        if total_role_skills else 0)



    missing_skills = [
                        original_skill
                        for normalized_skill, original_skill in role_index.items()
                        if normalized_skill not in matched_role_skills
    ]

    return {
            "coverage_score": round(coverage_score, 2)*100,
            "matched_count": len(matched_role_skills),

            "total_role_skills": total_role_skills,
            "matched": matched_pairs,
            "missing_skills": missing_skills
    }

  def semantic_match_proj(self,role_embedding):


    if not self.project_text.strip():
      return {"score":0}

    similarity = cosine_similarity(
                                    [self.project_embedding],
                                    [role_embedding]
    )[0][0]*100



    return {"score":similarity}

  def semantic_match_exp(self,role_embedding):
    if not self.exp_text.strip():
      return {"score":0}


    similarity = cosine_similarity(
                                  [self.exp_embeddings],
                                  [role_embedding]
    )[0][0]*100
    return {"score":similarity}
  
  def split_bullets(self,project_text):
    bullets = re.split(r'\s*•\s*', project_text)

    bullets = [b.strip() for b in bullets if b.strip()]

    # Fallback if no real bullets detected
    if len(bullets) <= 1:
      bullets = [line.strip() for line in project_text.split("\n") if line.strip()]
    return bullets


  def get_skill_embedding(self,skill):

    primary = self.store["primary_skills"]
    secondary = self.store["secondary_skills"]

    if skill in primary["item_to_index"]:
        idx = primary["item_to_index"][skill]
        return primary["embeddings"][idx]

    if skill in secondary["item_to_index"]:
        idx = secondary["item_to_index"][skill]
        return secondary["embeddings"][idx]

    return None
  

  def extract_project_skills_hybrid(
    self,
    master_skill_list,
    fuzzy_threshold=70,
    semantic_threshold=0.5,
    inference_threshold=0.5
):
    if not self.project_text.strip():
      return {
        "skills": [],
        "missing": master_skill_list
    }

    skill_scores = {}   # skill -> {score, method}

    

    # -----------------------------------
    # DIRECT / FUZZY / SEMANTIC MATCH
    # -----------------------------------
    for skill in master_skill_list:

        normalized_skill = self.normalize_text(skill)

        # 1️⃣ DIRECT MATCH
        if re.search(rf'\b{re.escape(normalized_skill)}\b', self.normalized_project):

            score = 1.0
            method = "direct"

        else:

            # 2️⃣ FUZZY MATCH
            match, fuzzy_score = self.match_skill(
                skill,
                [self.normalized_project],
                fuzzy_threshold
            )

            if match:
                score = fuzzy_score / 100
                method = "fuzzy"

            else:
                # 3️⃣ SEMANTIC MATCH
                skill_emb = self.get_skill_embedding(skill)
                
                if skill_emb is None:
                  continue

                similarities = cosine_similarity(
                    [skill_emb],
                    self.bullet_embeddings
                )[0]

                max_similarity = similarities.max()

                if max_similarity >= semantic_threshold:
                    score = float(max_similarity)
                    method = "semantic"
                else:
                    continue

        # Keep highest score per skill
        if skill not in skill_scores or score > skill_scores[skill]["score"]:
            skill_scores[skill] = {
                "score": round(score, 3),
                "method": method
            }

    # -----------------------------------
    # SEMANTIC SKILL INFERENCE
    # -----------------------------------
    detected_skills = list(skill_scores.keys())

    for detected in detected_skills:

        detected_emb = self.get_skill_embedding(detected)

        for skill in master_skill_list:

            if skill in skill_scores:
                continue

            skill_emb = self.get_skill_embedding(skill)

            sim = cosine_similarity(
                [detected_emb],
                [skill_emb]
            )[0][0]

            if sim >= inference_threshold:

                inferred_score = sim * 0.4

                skill_scores[skill] = {
                    "score": round(inferred_score, 3),
                    "method": "inferred"
                }

    # -----------------------------------
    # FORMAT RESULTS
    # -----------------------------------
    found_skills = [
        {
            "skill": skill,
            "score": data["score"],
            "method": data["method"]
        }
        for skill, data in skill_scores.items()
    ]

    found_skill_set = set(skill_scores.keys())

    missing_skills = [
        skill for skill in master_skill_list
        if skill not in found_skill_set
    ]

    return {
        "skills": found_skills,
        "missing": missing_skills
    }


  def __init__(self,resume_skills,project_text,exp_text,achiev_text,cert_list,model,job_df,store):
    self.job_df=job_df
    self.store=store
    self.model=model
    self.resume_skills=resume_skills
    self.project_text=project_text
    self.bullets = self.split_bullets(self.project_text)
    self.normalized_project=self.normalize_text(self.project_text)
    self.exp_text=exp_text
    self.normalized_exp=self.normalize_text(self.exp_text)
    self.achiev_text=achiev_text
    self.cert_list=cert_list
    self.certificates=[]
    if self.cert_list:
      cert_embeddings = self.model.encode(self.cert_list)
      for cert, cert_emb in zip(self.cert_list, cert_embeddings):
        cert_sample={}
        cert_sample["text"]=cert
        cert_sample["embd"]=cert_emb
        self.certificates.append(cert_sample)
    self.ALIAS_MAP = {
      "ml": "machine learning",
      "js": "javascript",
      "nodejs": "node.js",
      "postgres": "postgresql",
      "ci cd": "continuous integration"
    }

    self.LEADERSHIP_PATTERN = re.compile(r"""
    \b(
    led|managed|mentored|supervised|owned|
    directed|spearheaded|captained|
    coordinated|headed|initiated|drove
    )\b
    """, re.IGNORECASE | re.VERBOSE)

    self.TEAM_PATTERN = re.compile(r"\b\d+\s?(-| )?member\b", re.IGNORECASE)
    self.TIER1_ORGS = [
                "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix",
                "OpenAI", "NVIDIA", "Tesla", "Adobe", "IBM", "Oracle",
                "McKinsey", "BCG", "Bain", "Deloitte", "PwC", "EY", "KPMG",
                "Goldman Sachs", "Morgan Stanley", "JPMorgan", "BlackRock",
                "IIT", "IISc", "IIM", "BITS Pilani",
                "Stanford", "MIT", "Harvard", "Oxford", "Cambridge",
                "Kaggle", "Codeforces", "LeetCode", "ICPC"
    ]

    self.COMPETITION_PATTERN = re.compile(r"""
    \b(
    hackathon|
    competition|
    contest|
    olympiad|
    challenge|
    champion|
    winner|
    won|
    finalist|
    runner\s?up|
    rank(ed)?\s?\d+|
    1st|2nd|3rd|
    top\s?\d+
    )\b
    """, re.IGNORECASE | re.VERBOSE)

    self.SCOPE_PATTERN = re.compile(r"""
    \b(
    international|
    global|
    national|
    state|
    regional|
    intercollege|
    college|
    university|
    institute
    )\b
    """, re.IGNORECASE | re.VERBOSE)

    self.RANK_PATTERN = re.compile(r"""
    \b(
    1st|2nd|3rd|
    winner|won|
    champion|
    finalist|
    runner\s?up|
    rank(ed)?\s?\d+|
    top\s?\d+
    )\b
    """, re.IGNORECASE | re.VERBOSE)

    self.TIER1_CERT_PROVIDERS = [
          "AWS", "Amazon Web Services",
          "Microsoft", "Azure",
          "Google Cloud", "GCP",
          "Cisco", "CCNA", "CCNP",
          "Red Hat", "RHCE",
          "Oracle",
          "CompTIA",
          "PMI", "PMP"

    ]

    self.TIER2_CERT_PROVIDERS = [
          "Coursera",
           "edX",
           "Udacity",
           "NPTEL",
            "Udemy"
    ]

    self.CERT_LEVEL_PATTERN = re.compile(
        r"\b(associate|professional|expert|architect|specialist|advanced)\b",
        re.IGNORECASE
    )



    self.QUANT_REGEX = r"""
\b(
    # Percentages
    \d+% |
    \d+\s?percent |

    # Time units
    \d+\s?(ms|millisecond(s)?|sec(ond)?s?|minute(s)?|hour(s)?) |

    # Money
    \$\s?\d+[kKmMbB]? |

    # Scale (k, m, b)
    \d+[kKmMbB]\+? |

    # Multiplier
    \d+x |

    # Ranges
    \d+\s*(to|-)\s*\d+ |

    # Rankings
    top\s?\d+ |
    rank(ed)?\s?\d+ |
    1st|2nd|3rd |

    # Quantity with + and descriptors
    \d+\+?\s?(active\s)?(users|clients|projects|models|teams|registrations|requests|events) |

    # Improvements / reductions
    (improved|reduced|increased|optimized|boosted)\s+\d+% |

    # Uptime
    \d+%\s?(uptime|availability)
)
\b
"""
    resume_parts = []
    resume_parts.extend(self.bullets)
    resume_parts.append(self.normalized_project)
    resume_parts.append(self.normalized_exp)
    resume_parts.append(self.achiev_text)
    embeddings = self.model.encode(resume_parts) if resume_parts else []
    bullet_count = len(self.bullets)
    self.bullet_embeddings = embeddings[:bullet_count]
    self.project_embedding = embeddings[bullet_count] if len(embeddings) > bullet_count else None
    self.exp_embeddings = embeddings[bullet_count + 1] if len(embeddings) > bullet_count + 1 else None
    self.ach_emb = embeddings[bullet_count + 2] if len(embeddings) > bullet_count + 2 else None
    self.strong_emb = self.store["anchors"]["embeddings"][
        self.store["anchors"]["item_to_index"]["STRONG_IMPACT_ANCHOR"]
      ]
    self.weak_emb = self.store["anchors"]["embeddings"][
        self.store["anchors"]["item_to_index"]["WEAK_IMPACT_ANCHOR"]
      ]
    self.leadership_emb = self.store["anchors"]["embeddings"][
        self.store["anchors"]["item_to_index"]["LEADERSHIP_ANCHOR"]
      ]
    self.prestige_emb = self.store["anchors"]["embeddings"][
        self.store["anchors"]["item_to_index"]["PRESTIGE_ANCHOR"]
      ]


  def compute_certificate_score(self, jd_emb):
    """
    certificates_list : list of certificate strings
    jd_text           : job description text
    Returns score between 0–100
    """

    if not self.certificates:
      return {"final_score":0}


    cert_scores = []

    for cert in self.certificates:



      # -----------------------------
      # 1️⃣ Relevance (Embedding-Based)
      # -----------------------------
      relevance = cosine_similarity([cert["embd"]], [jd_emb])[0][0] * 100
      relevance = max(0, min(relevance, 100))

      # -----------------------------
      # 2️⃣ Authority Score
      # -----------------------------
      text_lower = cert["text"].lower()
      authority = 50  # default base

      for provider in self.TIER1_CERT_PROVIDERS:
        if provider.lower() in text_lower:
          authority = 90
          break
        else:
          for provider in self.TIER2_CERT_PROVIDERS:
            if provider.lower() in text_lower:
              authority = 70
              break

      # -----------------------------
      # 3️⃣ Level Score
      # -----------------------------
      if self.CERT_LEVEL_PATTERN.search(cert["text"]):
        level = 85
      else:
        level = 60

      # -----------------------------
      # 4️⃣ Combined Certificate Score
      # -----------------------------
      score = (
              0.5 * relevance +
              0.3 * authority +
              0.2 * level
      )

      score = max(0, min(score, 100))
      cert_scores.append(score)

    # -----------------------------
    # 5️⃣ Anti-Gaming: Top 2 Only
    # -----------------------------
    cert_scores.sort(reverse=True)
    top_scores = cert_scores[:2]

    final_score = sum(top_scores) / len(top_scores)

    return {
        "final_score":round(final_score, 2)
        }

  def compute_prestige_score(self, achievements_text):

    text = achievements_text.lower()

    # ---- Base semantic prestige ----
    prestige_score = cosine_similarity(
                                      [self.ach_emb],
                                      [self.prestige_emb]
    )[0][0] * 100


    # ---- Competition signal ----
    if self.COMPETITION_PATTERN.search(text):
      prestige_score += 20

    # ---- Ranking signal ----
    if self.RANK_PATTERN.search(text):
      prestige_score += 20

    # ---- Scope signal ----
    if "international" in text or "global" in text:
      prestige_score += 25
    elif "national" in text:
      prestige_score += 20
    elif "state" in text or "regional" in text:
      prestige_score += 15
    elif "college" in text or "university" in text:
      prestige_score += 10

    # ---- Recognized org signal ----
    for org in self.TIER1_ORGS:
      if org.lower() in text:
        prestige_score += 20
        break

    # ---- Clamp ----
    prestige_score = min(max(prestige_score, 0), 100)

    return prestige_score

  def competition_bonus(self,text):
    text_lower = text.lower()

    if not self.COMPETITION_PATTERN.search(text_lower):
      return 0

    # Base competition bonus
    bonus = 20

    # Scope-based adjustment
    if "international" in text_lower or "global" in text_lower:
      bonus = 50
    elif "national" in text_lower:
      bonus = 30
    elif "state" in text_lower or "regional" in text_lower:
      bonus = 25
    else:
      bonus = 20

    return bonus

  def compute_achievement_score(self,jd_emb):


    """
    Computes achievement score (0–100) using:
    - Semantic anchors
    - Tier-1 org detection
    - Quantification regex
    - JD relevance
    """

    
    

    
    if(self.achiev_text == ""):
      return {
            "final_score": 0,
            "semantic_impact": 0,
            "relevance": 0,
            "leadership": 0,
            "prestige": 0,
            "comp_bonus": 0,
            "quant_bonus": 0
    }


    comp_bonus = self.competition_bonus(text=self.achiev_text)





    strong_score = cosine_similarity(
                            [self.ach_emb],
                            [self.strong_emb]
    )[0][0]

    weak_score = cosine_similarity(
                                    [self.ach_emb],
                                    [self.weak_emb]
    )[0][0]


    semantic_raw = strong_score - weak_score
    semantic_impact_score = 100 * (1 / (1 + np.exp(-5 * semantic_raw)))


    leadership_score = cosine_similarity(
                                        [self.ach_emb],
                                        [self.leadership_emb]
    )[0][0] * 100

    if self.LEADERSHIP_PATTERN.search(self.achiev_text):
      leadership_score += 25

    if self.TEAM_PATTERN.search(self.achiev_text):
      leadership_score += 15

    leadership_score = min(leadership_score, 100)


    prestige_score = self.compute_prestige_score(achievements_text=self.achiev_text)




    quant_bonus = 0
    if re.search(self.QUANT_REGEX, self.achiev_text, re.IGNORECASE | re.VERBOSE):
      quant_bonus = 25

    relevance_score = cosine_similarity(
                                        [self.ach_emb],
                                        [jd_emb]
    )[0][0] * 100


    final_score = (
                    0.35 * semantic_impact_score +
                    0.35 * relevance_score +
                    0.2 * comp_bonus +
                    0.15 * leadership_score +
                    0.15 * prestige_score +
                    0.2 * quant_bonus
    )

    final_score = max(0, min(final_score, 100))

    return {
            "final_score": round(final_score, 2),
            "semantic_impact": round(semantic_impact_score, 2),
            "relevance": round(relevance_score, 2),
            "leadership": round(leadership_score, 2),
            "prestige": round(prestige_score, 2),
            "comp_bonus": comp_bonus,
            "quant_bonus": quant_bonus
    }

  def final_score(self,exp_year):
    result=[]

    for _ , row in self.job_df.iterrows():
      
      if not (row['MinYears'] <= exp_year):
        continue
      skill_result_primary=self.compare_skill_lists(role_skills=row['PrimarySkills'],threshold=70)
      skill_result_secondry=self.compare_skill_lists(role_skills=row['SecondarySkills'],threshold=70)
      role_name = row["Title"]
      jd_embd=self.store["job_text"]["embeddings"][
        self.store["job_text"]["item_to_index"][role_name]
      ]
      project_result=self.semantic_match_proj(role_embedding=jd_embd)
      all_skills=row['PrimarySkills']+row['SecondarySkills']
      project_skill_result=self.extract_project_skills_hybrid(master_skill_list=all_skills)
      project_skills=project_skill_result["skills"]
      missing=project_skill_result["missing"]
      project_skill_score=(len(project_skills)/len(all_skills)*100) if all_skills else 0
      exp_result=self.semantic_match_exp(role_embedding=jd_embd)
      achiev_result=self.compute_achievement_score(jd_emb=jd_embd)
      cert_result=self.compute_certificate_score(jd_emb=jd_embd)
      project_final_score=(0.6*project_result["score"])+(0.4*project_skill_score)
      new_result={}
      new_result["Title"]=role_name
      if exp_year<1:
        new_result["score"]=(
          0.25*project_final_score +
          0.3*skill_result_primary["coverage_score"] +
          0.15*skill_result_secondry["coverage_score"] +
          0.15*achiev_result["final_score"] +
          0.05*cert_result["final_score"] +
          0.1*exp_result["score"]
        )
      else:
        new_result["score"]=(

                    0.35*skill_result_primary["coverage_score"] +
                    0.15*skill_result_secondry["coverage_score"] +
                    0.15*achiev_result["final_score"] +
                    0.1*cert_result["final_score"] +
                    0.25*exp_result["score"]
        )
      new_result["Responsibilities"]=row["Responsibilities"]
      new_result["primary_skill"]=skill_result_primary
      new_result["secondry_skill"]=skill_result_secondry
      new_result["projects"]={
        "semantic_score": project_result["score"],
        "skills": [item["skill"] for item in project_skills],
        "missing": missing,
        "match_score": project_skill_score,
        "final_score":project_final_score
      }
      new_result["experience"]=exp_result
      new_result["achievment"]=achiev_result
      new_result["certificates"]=cert_result
      
      result.append(new_result)
      

    result=sorted(
            result,
            key= lambda x: x['score'],
            reverse=True
    )
    return result[:5]


