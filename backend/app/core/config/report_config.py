from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class TeamType(Enum):
    MARKETING = "marketing"
    PURCHASING = "purchasing"
    DEVELOPMENT = "development"
    GENERAL_AFFAIRS = "general_affairs"
    GENERAL = "general"

class ReportType(Enum):
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

class Language(Enum):
    KOREAN = "korean"
    ENGLISH = "english"

@dataclass
class ChartConfig:
    type: str
    title: str
    description: str

@dataclass
class SectionConfig:
    key: str
    words: str
    details: List[str]
    chart_requirements: List[str]

@dataclass
class ReportTemplate:
    role_description: str
    sections: List[SectionConfig]
    total_words: str
    charts: str
    
