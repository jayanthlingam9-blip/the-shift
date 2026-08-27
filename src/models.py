from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Domain(StrEnum):
    legal = "legal"
    finance = "finance"
    medical = "medical"


class SourceType(StrEnum):
    pdf = "pdf"
    html = "html"
    csv = "csv"


class SourceDocument(BaseModel):
    document_id: str
    domain: Domain
    source_type: SourceType
    path: Path


class NormalizedElement(BaseModel):
    element_id: str
    document_id: str
    domain: Domain
    element_type: str
    text: str
    metadata: dict = Field(default_factory=dict)


class ParentSection(BaseModel):
    parent_id: str
    document_id: str
    domain: Domain
    source_type: SourceType
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    parent_text: str = ""
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    parent_id: str | None = None
    document_id: str
    domain: Domain
    source_type: SourceType
    file_name: str
    section_title: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    text_content: str | None = None
    table_markdown: str | None = None
    table_html: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    label: str
    metadata: dict = Field(default_factory=dict)
