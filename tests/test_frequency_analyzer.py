# tests/test_frequency_analyzer.py
"""Тесты частотного анализатора."""
from __future__ import annotations

import pytest

from src.analysis.frequency_analyzer import FrequencyAnalyzer


@pytest.fixture
def analyzer() -> FrequencyAnalyzer:
    return FrequencyAnalyzer()


class TestAnalyzeFrequency:
    def test_basic(self, analyzer: FrequencyAnalyzer):
        lemmas = ["кот", "кот", "собака"]
        result = analyzer.analyze_frequency(lemmas, total_words=100)

        assert len(result) == 2
        words = [r[0] for r in result]
        assert "кот" in words
        assert "собака" in words

    def test_sorted_by_ipm_descending(self, analyzer: FrequencyAnalyzer):
        lemmas = ["а", "б", "б", "в", "в", "в"]
        result = analyzer.analyze_frequency(lemmas, total_words=6)
        ipms   = [r[2] for r in result]
        assert ipms == sorted(ipms, reverse=True)

    def test_ipm_calculation(self, analyzer: FrequencyAnalyzer):
        """ipm = (freq / total_words) * 1_000_000"""
        lemmas = ["кот", "кот"]  # freq=2, total=1000
        result = analyzer.analyze_frequency(lemmas, total_words=1000)
        word, abs_freq, ipm = result[0]
        assert word     == "кот"
        assert abs_freq == 2
        assert ipm      == pytest.approx(2000.0)

    def test_zero_total_words(self, analyzer: FrequencyAnalyzer):
        """При total_words=0 ipm должен быть 0, не ZeroDivisionError."""
        result = analyzer.analyze_frequency(["кот"], total_words=0)
        assert result[0][2] == 0.0

    def test_empty_lemmas(self, analyzer: FrequencyAnalyzer):
        result = analyzer.analyze_frequency([], total_words=100)
        assert result == []

    def test_single_word(self, analyzer: FrequencyAnalyzer):
        result = analyzer.analyze_frequency(["слово"], total_words=1)
        assert len(result) == 1
        assert result[0][1] == 1           # abs_freq
        assert result[0][2] == 1_000_000.0  # ipm

    def test_returns_list_of_tuples(self, analyzer: FrequencyAnalyzer):
        result = analyzer.analyze_frequency(["кот"], total_words=10)
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 3

    def test_abs_freq_matches_count(self, analyzer: FrequencyAnalyzer):
        lemmas = ["а"] * 5 + ["б"] * 3 + ["в"] * 1
        result = analyzer.analyze_frequency(lemmas, total_words=100)
        freq_map = {word: freq for word, freq, _ in result}
        assert freq_map["а"] == 5
        assert freq_map["б"] == 3
        assert freq_map["в"] == 1


class TestToDataframe:
    def test_columns(self, analyzer: FrequencyAnalyzer):
        freq_list = [("кот", 2, 100.0), ("пёс", 1, 50.0)]
        df        = analyzer.to_dataframe(freq_list)
        assert list(df.columns) == [
            "Слово", "Сумма слов", "Нормализованная частота"
        ]

    def test_values(self, analyzer: FrequencyAnalyzer):
        freq_list = [("кот", 5, 500.0)]
        df        = analyzer.to_dataframe(freq_list)
        assert df.iloc[0]["Слово"]                   == "кот"
        assert df.iloc[0]["Сумма слов"]              == 5
        assert df.iloc[0]["Нормализованная частота"] == 500.0

    def test_empty_list(self, analyzer: FrequencyAnalyzer):
        df = analyzer.to_dataframe([])
        assert len(df) == 0
        assert "Слово" in df.columns