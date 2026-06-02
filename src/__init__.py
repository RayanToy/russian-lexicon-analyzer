# src/__init__.py
from src.preprocessing.text_preprocessor import (
    StandardTextPreprocessor,
    TextPreprocessorWithoutStopwords,
)
from src.analysis.frequency_analyzer import FrequencyAnalyzer
from src.analysis.file_processor import FileProcessor
from src.analysis.frequency_aggregator import FrequencyAggregator
from src.analysis.data_analyzer import DataAnalyzer
from src.highlighting.lexicon import LexiconRepository
from src.highlighting.highlighter import highlight_docx

__all__ = [
    "StandardTextPreprocessor",
    "TextPreprocessorWithoutStopwords",
    "FrequencyAnalyzer",
    "FileProcessor",
    "FrequencyAggregator",
    "DataAnalyzer",
    "LexiconRepository",
    "highlight_docx",
]