# tests/test_lexicon.py
"""Тесты модуля lexicon: нормализация, загрузка лексикона, стоп-слова."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.highlighting.lexicon import (
    LemmaNormalizer,
    LexiconRepository,
    load_stopword_file,
)


class TestLemmaNormalizer:
    @pytest.mark.parametrize("inp, expected", [
        ("Кот",      "кот"),
        ("СОБАКА",   "собака"),
        ("  пёс  ",  "пес"),      # ё → е и strip
        ("ЁЖ",       "еж"),
        ("",         ""),
        ("  ",       ""),
    ])
    def test_normalize(self, inp: str, expected: str):
        assert LemmaNormalizer.normalize(inp) == expected

    def test_normalize_none_safe(self):
        """normalize не должен падать на None-подобных значениях."""
        assert LemmaNormalizer.normalize("") == ""

    def test_normalize_yo_replacement(self):
        assert LemmaNormalizer.normalize("ёжик") == "ежик"
        assert LemmaNormalizer.normalize("Ёлка") == "елка"


class TestLexiconRepository:
    def test_load_from_xlsx(self, tmp_xlsx: Path):
        repo   = LexiconRepository()
        lemmas = repo.load_lemmas(tmp_xlsx)
        assert isinstance(lemmas, set)
        assert "кот" in lemmas
        assert "собака" in lemmas

    def test_load_from_csv(self, tmp_csv: Path):
        repo   = LexiconRepository()
        lemmas = repo.load_lemmas(tmp_csv)
        assert "кот" in lemmas

    def test_custom_column_name(self, tmp_path: Path):
        path = tmp_path / "lex.xlsx"
        df   = pd.DataFrame({"lemma": ["слово", "текст"]})
        df.to_excel(path, index=False)

        repo   = LexiconRepository(lemma_column_name="lemma")
        lemmas = repo.load_lemmas(path)
        assert "слово" in lemmas
        assert "текст" in lemmas

    def test_raises_on_missing_column(self, tmp_path: Path):
        path = tmp_path / "bad.xlsx"
        df   = pd.DataFrame({"wrong_col": ["a", "b"]})
        df.to_excel(path, index=False)

        repo = LexiconRepository(lemma_column_name="Слово")
        with pytest.raises(ValueError, match="не найден"):
            repo.load_lemmas(path)

    def test_raises_on_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.xlsx"
        pd.DataFrame().to_excel(path, index=False)
        with pytest.raises(ValueError, match="пуст"):
            LexiconRepository().load_lemmas(path)

    def test_lemmas_are_normalized(self, tmp_path: Path):
        """Все леммы должны быть нормализованы: lower + ё→е."""
        path = tmp_path / "lex.xlsx"
        df   = pd.DataFrame({"Слово": ["Кот", "ЁЖИК", "  Пёс  "]})
        df.to_excel(path, index=False)

        lemmas = LexiconRepository().load_lemmas(path)
        assert "кот"  in lemmas
        assert "ежик" in lemmas
        assert "пес"  in lemmas

    def test_skips_nan_values(self, tmp_path: Path):
        path = tmp_path / "with_nan.xlsx"
        df   = pd.DataFrame({"Слово": ["кот", None, "собака"]})
        df.to_excel(path, index=False)

        lemmas = LexiconRepository().load_lemmas(path)
        assert "nan" not in lemmas
        assert "кот" in lemmas


class TestLoadStopwordFile:
    def test_loads_words(self, tmp_stopwords_file: Path):
        words = load_stopword_file(tmp_stopwords_file)
        assert isinstance(words, set)
        assert "также"   in words
        assert "однако"  in words
        assert "поэтому" in words

    def test_returns_empty_set_if_file_missing(self, tmp_path: Path):
        result = load_stopword_file(tmp_path / "nonexistent.txt")
        assert result == set()

    def test_normalizes_words(self, tmp_path: Path):
        path = tmp_path / "sw.txt"
        path.write_text("Ёжик\nКОТ\n  пёс  \n", encoding="utf-8")
        words = load_stopword_file(path)
        assert "ежик" in words
        assert "кот"  in words
        assert "пес"  in words

    def test_skips_empty_lines(self, tmp_path: Path):
        path = tmp_path / "sw.txt"
        path.write_text("слово\n\n\nдругое\n", encoding="utf-8")
        words = load_stopword_file(path)
        assert "" not in words
        assert len(words) == 2