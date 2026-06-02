# src/highlighting/lexicon.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set

from src.preprocessing.utils import Utils


class LemmaNormalizer:
    """
    Нормализует леммы для единообразного сравнения.

    Правила нормализации:
    - strip() — убираем пробелы по краям
    - lower() — приводим к нижнему регистру
    - ё → е   — унифицируем написание

    Все компоненты системы должны нормализовывать леммы
    через этот класс, чтобы избежать расхождений.
    """

    @staticmethod
    def normalize(s: str) -> str:
        return (s or "").strip().lower().replace('ё', 'е')


class LexiconRepository:
    """
    Загружает лексикон из XLSX или CSV файла.

    Лексикон — множество нормализованных лемм, с которыми
    сравниваются токены из текста. Слова из лексикона
    считаются «известными» и не подсвечиваются.

    Args:
        lemma_column_name: название колонки с леммами.
                           Если None — используется первая колонка файла.
    """

    def __init__(self, lemma_column_name: Optional[str] = None) -> None:
        self.lemma_column_name = lemma_column_name

    def load_lemmas(self, file_path: str | Path) -> Set[str]:
        """
        Загружает множество нормализованных лемм из файла.

        Args:
            file_path: путь к XLSX или CSV файлу с лексиконом

        Returns:
            Множество нормализованных лемм (str)

        Raises:
            ValueError: если файл пуст или колонка не найдена
            FileNotFoundError: если файл не существует
        """
        df = Utils.load_dataframe(file_path)

        if df.empty:
            raise ValueError(f"Файл лексикона пуст: {file_path}")

        lemma_col = self.lemma_column_name or df.columns[0]
        if lemma_col not in df.columns:
            raise ValueError(
                f"Колонка '{lemma_col}' не найдена в файле {file_path}.\n"
                f"Доступные колонки: {list(df.columns)}"
            )

        return {
            LemmaNormalizer.normalize(x)
            for x in df[lemma_col].astype(str).tolist()
            if str(x).strip() and str(x).strip().lower() != 'nan'
        }


def load_stopword_file(path: str | Path) -> Set[str]:
    """
    Загружает множество стоп-слов из текстового файла (одно слово на строку).

    Args:
        path: путь к файлу стоп-слов

    Returns:
        Множество нормализованных стоп-слов.
        Если файл не существует — возвращает пустое множество.
    """
    path = Path(path)
    if not path.exists():
        return set()

    stopset: Set[str] = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                stopset.add(LemmaNormalizer.normalize(word))
    return stopset