# src/analysis/frequency_analyzer.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from nltk import FreqDist


# Тип для одной записи частотного списка: (слово, абс. частота, норм. частота)
FreqEntry = Tuple[str, int, float]


class FrequencyAnalyzer:
    """
    Вычисляет частотные характеристики списка лемм.

    Нормализованная частота считается в ipm (instances per million) —
    стандартная единица в корпусной лингвистике.
    """

    def analyze_frequency(
        self,
        lemmas: List[str],
        total_words: int,
    ) -> List[FreqEntry]:
        """
        Строит частотный список из набора лемм.

        Args:
            lemmas: список лемм после предобработки
            total_words: общий объём текста в словах
                         (знаменатель для нормализации — может быть больше
                          len(lemmas), т.к. часть слов отфильтрована)

        Returns:
            Список кортежей (слово, абс_частота, ipm),
            отсортированный по убыванию ipm.

        Example:
            >>> analyzer = FrequencyAnalyzer()
            >>> analyzer.analyze_frequency(["кот", "кот", "собака"], 100)
            [("кот", 2, 20000.0), ("собака", 1, 10000.0)]
        """
        if not lemmas:
            return []

        fdist = FreqDist(lemmas)
        ipm_factor = 1_000_000 / total_words if total_words > 0 else 0.0

        result = [
            (word, freq, round(freq * ipm_factor, 2))
            for word, freq in fdist.items()
        ]
        return sorted(result, key=lambda x: x[2], reverse=True)

    def to_dataframe(self, freq_list: List[FreqEntry]) -> pd.DataFrame:
        """
        Конвертирует частотный список в DataFrame.

        Args:
            freq_list: результат analyze_frequency()

        Returns:
            DataFrame с колонками: Слово, Сумма слов, Нормализованная частота
        """
        return pd.DataFrame(
            freq_list,
            columns=['Слово', 'Сумма слов', 'Нормализованная частота'],
        )