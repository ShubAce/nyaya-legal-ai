"""
schema.py — Canonical extraction schema for Indian legal judgments.

This is the single source of truth for what Nyaya-7B is trained to produce.
Every training sample, evaluation metric, and agent output references this schema.
"""

from typing import Optional
from pydantic import BaseModel, field_validator


# ── Valid values for controlled fields ────────────────────────────────────────

VALID_COURTS = {
    "Supreme Court of India",
    "High Court of Allahabad",
    "High Court of Bombay",
    "High Court of Calcutta",
    "High Court of Delhi",
    "High Court of Gujarat",
    "High Court of Karnataka",
    "High Court of Kerala",
    "High Court of Madras",
    "High Court of Patna",
    "High Court of Punjab and Haryana",
    "High Court of Rajasthan",
    "High Court of Andhra Pradesh",
    "High Court of Telangana",
    "High Court of Jharkhand",
    "High Court of Chhattisgarh",
    "High Court of Uttarakhand",
    "High Court of Himachal Pradesh",
    "High Court of Manipur",
    "High Court of Meghalaya",
    "High Court of Sikkim",
    "High Court of Tripura",
    "High Court of Gauhati",
    "High Court of Orissa",
    "National Consumer Disputes Redressal Commission",
    "National Green Tribunal",
    "Armed Forces Tribunal",
}

VALID_OUTCOMES = {
    "allowed",       # appeal/petition granted
    "dismissed",     # appeal/petition rejected
    "disposed",      # disposed of with directions
    "remanded",      # sent back to lower court
    "modified",      # partially allowed
    "withdrawn",     # withdrawn by petitioner
    "settled",       # settled out of court
}

VALID_SUBJECT_MATTERS = {
    "Criminal",
    "Civil",
    "Constitutional",
    "Service",
    "Tax",
    "Family",
    "Property",
    "Labour",
    "Company",
    "Environmental",
    "Consumer",
    "Arbitration",
    "Intellectual Property",
    "Election",
    "Administrative",
}

KNOWN_ACTS = {
    "Indian Penal Code",
    "Code of Criminal Procedure",
    "Code of Civil Procedure",
    "Indian Evidence Act",
    "Constitution of India",
    "Income Tax Act",
    "Companies Act",
    "Transfer of Property Act",
    "Specific Relief Act",
    "Contract Act",
    "Limitation Act",
    "Negotiable Instruments Act",
    "Motor Vehicles Act",
    "Prevention of Corruption Act",
    "Narcotic Drugs and Psychotropic Substances Act",
    "Protection of Women from Domestic Violence Act",
    "Hindu Marriage Act",
    "Hindu Succession Act",
    "Muslim Personal Law",
    "Arbitration and Conciliation Act",
    "Consumer Protection Act",
    "Right to Information Act",
    "Environment Protection Act",
    "Forest Conservation Act",
    "Land Acquisition Act",
    "Industrial Disputes Act",
}

# ── Pydantic models ───────────────────────────────────────────────────────────

class StatuteCitation(BaseModel):
    act: str
    section: str
    description: Optional[str] = None

    @field_validator("act")
    @classmethod
    def normalize_act_name(cls, v: str) -> str:
        # Normalize common abbreviations
        abbrevs = {
            "IPC": "Indian Penal Code",
            "CrPC": "Code of Criminal Procedure",
            "CPC": "Code of Civil Procedure",
            "IEA": "Indian Evidence Act",
        }
        return abbrevs.get(v.strip(), v.strip())


class PrecedentCitation(BaseModel):
    citation: str          # e.g. "AIR 1984 SC 1622"
    case_name: Optional[str] = None

    @field_validator("citation")
    @classmethod
    def validate_citation_format(cls, v: str) -> str:
        """Accept AIR, SCC, SCR, Cri LJ, etc. formats."""
        v = v.strip()
        return v


class LegalExtractionOutput(BaseModel):
    """
    The canonical output schema for Nyaya-7B.
    Every field matches a training label and an evaluation metric.
    """
    case_name: str
    citation: Optional[str] = None           # e.g. "AIR 1997 SC 3986"
    court: str
    bench: Optional[list[str]] = None        # list of judge names
    year: Optional[int] = None
    petitioner: str
    respondent: str
    subject_matter: Optional[str] = None
    statutes_cited: list[StatuteCitation]
    precedents_cited: list[PrecedentCitation]
    legal_issues: list[str]
    holding: str
    outcome: str
    outcome_label: Optional[int] = None      # 0=dismissed, 1=allowed, 2=disposed, etc.

    @field_validator("outcome")
    @classmethod
    def normalize_outcome(cls, v: str) -> str:
        v = v.lower().strip()
        # Map common variants
        mapping = {
            "appeal dismissed": "dismissed",
            "petition dismissed": "dismissed",
            "appeal allowed": "allowed",
            "petition allowed": "allowed",
            "disposed of": "disposed",
            "set aside and remanded": "remanded",
        }
        return mapping.get(v, v)

    @field_validator("court")
    @classmethod
    def normalize_court(cls, v: str) -> str:
        return v.strip().title() if v else v


# Outcome → integer label mapping (for classification metrics)
OUTCOME_TO_LABEL = {
    "dismissed": 0,
    "allowed": 1,
    "disposed": 2,
    "remanded": 3,
    "modified": 4,
    "withdrawn": 5,
    "settled": 6,
}

LABEL_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_LABEL.items()}


# ── Schema string for prompting Gemini 1.5 Flash labeller ─────────────────────────

SCHEMA_DESCRIPTION = """
{
  "case_name": "<Petitioner v. Respondent>",
  "citation": "<AIR/SCC/SCR citation if mentioned, else null>",
  "court": "<Full court name>",
  "bench": ["<Judge 1 name with J.>", "<Judge 2 name>"],
  "year": <4-digit year as integer>,
  "petitioner": "<Name of petitioner/appellant>",
  "respondent": "<Name of respondent>",
  "subject_matter": "<Criminal|Civil|Constitutional|Service|Tax|Family|...>",
  "statutes_cited": [
    {"act": "<Full act name>", "section": "<section number>", "description": "<brief description>"}
  ],
  "precedents_cited": [
    {"citation": "<AIR/SCC citation>", "case_name": "<case name if mentioned>"}
  ],
  "legal_issues": ["<Issue 1 as a question>", "<Issue 2>"],
  "holding": "<The court's actual decision and reasoning in 1-3 sentences>",
  "outcome": "<dismissed|allowed|disposed|remanded|modified|withdrawn|settled>",
  "outcome_label": <0=dismissed|1=allowed|2=disposed|3=remanded|4=modified|5=withdrawn|6=settled>
}
"""
