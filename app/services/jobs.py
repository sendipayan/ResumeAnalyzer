import urllib.parse


class JobRedirectBuilder:
    def __init__(self, role: str, skills: list, experience_years: int, location="India"):
        self.role = role.strip()
        self.skills = skills or []
        self.experience_years = experience_years
        self.location = location

    # -----------------------------
    # 1️⃣ Adjust Role Based on Experience
    # -----------------------------
    def _adjust_role_by_experience(self):
        role = self.role

        if self.experience_years <= 1:
            return f"Junior {role}"
        elif 2 <= self.experience_years <= 4:
            return role
        else:
            return f"Senior {role}"

    # -----------------------------
    # 2️⃣ Build Optimized Search Query
    # -----------------------------
    def build_query(self):
        adjusted_role = self._adjust_role_by_experience()

        # Take top 3 meaningful skills
        top_skills = [s for s in self.skills if len(s) > 2][:3]

        query_parts = [f'"{adjusted_role}"']
        query_parts.extend(top_skills)
        query_parts.append(self.location)

        final_query = " ".join(query_parts)
        return final_query

    # -----------------------------
    # 3️⃣ Generate Redirect URLs
    # -----------------------------
    def generate_links(self):
        query = self.build_query()
        encoded_query = urllib.parse.quote_plus(query)

        links = {
            "optimized_query": query,

            "google_jobs": (
                f"https://www.google.com/search?q={encoded_query}+jobs&ibp=htl;jobs"
            ),

            "linkedin": (
                f"https://www.linkedin.com/jobs/search/?keywords={encoded_query}&location={urllib.parse.quote_plus(self.location)}"
            ),

            "unstop": (
                f"https://unstop.com/jobs?search={encoded_query}"
            ),

            "wellfound": (
                f"https://wellfound.com/jobs?query={encoded_query}"
            )
        }

        return links


# -----------------------------
# 4️⃣ Example Usage
# -----------------------------
if __name__ == "__main__":
    predicted_role = ".NET Developer"
    predicted_skills = [
        "C#", "ASP.NET", "SQL Server", "Entity Framework", "LINQ"
    ]
    experience = 2

    builder = JobRedirectBuilder(
        role=predicted_role,
        skills=predicted_skills,
        experience_years=experience,
        location="India"
    )

    links = builder.generate_links()

    print("\n🔎 Optimized Search Query:")
    print(links["optimized_query"])

    print("\n🌍 Redirect Links:")
    for platform, url in links.items():
        if platform != "optimized_query":
            print(f"{platform}: {url}")