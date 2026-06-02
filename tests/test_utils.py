# tests/test_utils.py
"""Тесты вспомогательных функций."""
from __future__ import annotations

import string
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from src.preprocessing.utils import Utils


class TestRemoveCharsFromText:
    def test_removes_specified_chars(self):
        result = Utils.remove_chars_from_text("hello, world!", ",!")
        assert result == "hello  world "

    def test_replaces_with_space(self):
        """Символы заменяются пробелом, не удаляются."""
        result = Utils.remove_chars_from_text("a.b", ".")
        assert result == "a b"

    def test_empty_string(self):
        assert Utils.remove_chars_from_text("", ".,") == ""

    def test_no_chars_to_remove(self):
        assert Utils.remove_chars_from_text("hello", "") == "hello"

    def test_all_chars_removed(self):
        result = Utils.remove_chars_from_text("...", ".")
        assert result == "   "

    def test_unicode_chars(self):
        result = Utils.remove_chars_from_text("привет•мир", "•")
        assert result == "привет мир"


class TestLoadDataframe:
    def test_loads_xlsx(self, tmp_xlsx: Path):
        df = Utils.load_dataframe(tmp_xlsx)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Слово" in df.columns

    def test_loads_csv(self, tmp_csv: Path):
        df = Utils.load_dataframe(tmp_csv)
        assert isinstance(df, pd.DataFrame)
        assert "Слово" in df.columns

    def test_raises_on_unsupported_format(self, tmp_path: Path):
        bad_file = tmp_path / "data.json"
        bad_file.write_text("{}")
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            Utils.load_dataframe(bad_file)

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Utils.load_dataframe(tmp_path / "nonexistent.xlsx")


class TestMergeDataframes:
    def test_basic_merge(self):
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df2 = pd.DataFrame({"key": [1, 2],   "value": [10, 20]})
        result = Utils.merge_dataframes(df1, df2, "id", "key")
        assert len(result) == 3
        # Строка 3 не имеет пары в df2 → value = 0
        assert result.loc[result["id"] == 3, "value"].iloc[0] == 0

    def test_numeric_fillna_with_zero(self):
        df1 = pd.DataFrame({"id": [1, 2]})
        df2 = pd.DataFrame({"id": [1],  "count": [5]})
        result = Utils.merge_dataframes(df1, df2, "id", "id")
        assert result.loc[result["id"] == 2, "count"].iloc[0] == 0

    def test_string_fillna_with_empty(self):
        df1 = pd.DataFrame({"id": [1, 2]})
        df2 = pd.DataFrame({"id": [1], "label": ["hello"]})
        result = Utils.merge_dataframes(df1, df2, "id", "id")
        assert result.loc[result["id"] == 2, "label"].iloc[0] == ""

    def test_key_column_rename(self):
        df1 = pd.DataFrame({"id": [1]})
        df2 = pd.DataFrame({"other_id": [1], "val": [99]})
        result = Utils.merge_dataframes(df1, df2, "id", "other_id")
        # После rename df2 не должно содержать 'other_id'
        assert "other_id" not in result.columns
        assert "val" in result.columns