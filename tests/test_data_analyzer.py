# tests/test_data_analyzer.py
"""Тесты DataAnalyzer: агрегация, лексическая разница, коэффициент Жуайна."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.data_analyzer import DataAnalyzer


@pytest.fixture
def analyzer() -> DataAnalyzer:
    return DataAnalyzer()


@pytest.fixture
def two_freq_files(tmp_path: Path):
    """Два файла частотных списков для тестирования лексической разницы."""
    f1 = tmp_path / "list1.xlsx"
    f2 = tmp_path / "list2.xlsx"
    pd.DataFrame({
        "Слово":      ["кот", "собака", "дом"],
        "Сумма слов": [10, 5, 8],
    }).to_excel(f1, index=False)
    pd.DataFrame({
        "Слово":      ["кот", "математика", "теорема"],
        "Сумма слов": [3, 15, 7],
    }).to_excel(f2, index=False)
    return f1, f2


class TestProcessExcelFile:
    def test_accumulates_word_counts(
        self, analyzer: DataAnalyzer, tmp_frequency_xlsx: Path
    ):
        analyzer._process_excel_file(tmp_frequency_xlsx)
        assert "математика" in analyzer.word_sums
        assert analyzer.word_sums["математика"] == 30

    def test_handles_missing_columns_gracefully(
        self, analyzer: DataAnalyzer, tmp_path: Path
    ):
        """Файл без нужных колонок не должен вызывать исключение."""
        bad = tmp_path / "bad.xlsx"
        pd.DataFrame({"wrong": [1, 2]}).to_excel(bad, index=False)
        analyzer._process_excel_file(bad)  # не должно упасть
        assert analyzer.word_sums == {}

    def test_skips_nan_rows(
        self, analyzer: DataAnalyzer, tmp_path: Path
    ):
        path = tmp_path / "with_nan.xlsx"
        pd.DataFrame({
            "Слово":      ["кот", None, "собака"],
            "Сумма слов": [5,    3,    None],
        }).to_excel(path, index=False)
        analyzer._process_excel_file(path)
        # None в слове — пропускаем; None в счётчике — пропускаем
        assert "кот" in analyzer.word_sums


class TestLexicalDifference:
    def test_finds_new_words(
        self,
        analyzer: DataAnalyzer,
        two_freq_files,
        tmp_path: Path,
    ):
        f1, f2 = two_freq_files
        result = analyzer.calculate_lexical_difference(f1, f2, output_folder=tmp_path)

        words = set(result["Слово"])
        # "математика" и "теорема" есть в f2, но не в f1
        assert "математика" in words
        assert "теорема"    in words
        # "кот" есть в обоих — не должен попасть в разницу
        assert "кот" not in words

    def test_saves_xlsx(
        self,
        analyzer: DataAnalyzer,
        two_freq_files,
        tmp_path: Path,
    ):
        f1, f2 = two_freq_files
        analyzer.calculate_lexical_difference(f1, f2, output_folder=tmp_path)
        assert (tmp_path / "lexical_difference.xlsx").exists()

    def test_returns_dataframe(
        self,
        analyzer: DataAnalyzer,
        two_freq_files,
        tmp_path: Path,
    ):
        f1, f2   = two_freq_files
        result   = analyzer.calculate_lexical_difference(f1, f2, output_folder=tmp_path)
        assert isinstance(result, pd.DataFrame)

    def test_empty_difference(
        self,
        analyzer: DataAnalyzer,
        tmp_path: Path,
    ):
        """Если f2 ⊆ f1 — разница пустая."""
        f1 = tmp_path / "big.xlsx"
        f2 = tmp_path / "small.xlsx"
        pd.DataFrame({"Слово": ["кот", "собака"], "Сумма слов": [1, 1]}).to_excel(f1, index=False)
        pd.DataFrame({"Слово": ["кот"],            "Сумма слов": [1]}).to_excel(f2, index=False)
        result = analyzer.calculate_lexical_difference(f1, f2, output_folder=tmp_path)
        assert len(result) == 0


class TestJuainCoefficient:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Слово":   ["кот", "кот", "собака"],
            "freq_A":  [100.0,  100.0, 50.0],
            "freq_B":  [100.0,  100.0, 200.0],
            "freq_C":  [100.0,  100.0, 10.0],
        })

    def test_adds_column(self):
        df     = self._make_df()
        result = DataAnalyzer.calculate_juain_coefficient(
            df,
            thematic_segments=["A", "B", "C"],
            col_names=["freq_A", "freq_B", "freq_C"],
            output_column_name="juain",
        )
        assert "juain" in result.columns

    def test_uniform_distribution_gives_100(self):
        """Равномерное распределение → коэффициент = 100."""
        df = pd.DataFrame({
            "Слово":  ["кот"],
            "freq_A": [100.0],
            "freq_B": [100.0],
            "freq_C": [100.0],
        })
        result = DataAnalyzer.calculate_juain_coefficient(
            df,
            thematic_segments=["A", "B", "C"],
            col_names=["freq_A", "freq_B", "freq_C"],
            output_column_name="juain",
        )
        assert result.iloc[0]["juain"] == pytest.approx(100.0, abs=0.01)

    def test_zero_frequencies_gives_minus_2(self):
        """Нулевые частоты → коэффициент = -2."""
        df = pd.DataFrame({
            "Слово":  ["кот"],
            "freq_A": [0.0],
            "freq_B": [0.0],
            "freq_C": [0.0],
        })
        result = DataAnalyzer.calculate_juain_coefficient(
            df,
            thematic_segments=["A", "B", "C"],
            col_names=["freq_A", "freq_B", "freq_C"],
            output_column_name="juain",
        )
        assert result.iloc[0]["juain"] == -2.0

    def test_does_not_modify_original(self):
        """Исходный DataFrame не изменяется."""
        df     = self._make_df()
        before = df.copy()
        DataAnalyzer.calculate_juain_coefficient(
            df, ["A", "B", "C"],
            ["freq_A", "freq_B", "freq_C"],
            "juain",
        )
        pd.testing.assert_frame_equal(df, before)

    def test_coefficient_range(self):
        """Коэффициент ≤ 100 для любых данных."""
        df = pd.DataFrame({
            "Слово":  ["слово"] * 5,
            "freq_A": np.random.uniform(0, 1000, 5),
            "freq_B": np.random.uniform(0, 1000, 5),
            "freq_C": np.random.uniform(0, 1000, 5),
        })
        result = DataAnalyzer.calculate_juain_coefficient(
            df, ["A", "B", "C"],
            ["freq_A", "freq_B", "freq_C"],
            "juain",
        )
        valid = result[result["juain"] > -1]["juain"]
        assert (valid <= 100).all()