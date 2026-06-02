# src/highlighting/__init__.py
from src.highlighting.lexicon import (
    LemmaNormalizer,
    LexiconRepository,
    load_stopword_file,
)
from src.highlighting.matcher import TokenLemmaMatcher
from src.highlighting.highlighter import DocxHighlighter, highlight_docx

__all__ = [
    "LemmaNormalizer",
    "LexiconRepository",
    "load_stopword_file",
    "TokenLemmaMatcher",
    "DocxHighlighter",
    "highlight_docx",
]