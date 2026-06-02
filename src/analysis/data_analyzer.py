# src/analysis/data_analyzer.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


class DataAnalyzer:
    """
    Инструменты для сравнения и анализа частотных списков.

    Основные задачи:
    - агрегация по классу обучения
    - вычисление лексической разницы между двумя корпусами
    - расчёт коэффициента Жуайна (равномерность распределения по сегментам)
    """

    def __init__(self) -> None:
        self.word_sums: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Агрегация по классу
    # ------------------------------------------------------------------
    def process_folder(
        self,
        root_folder: Path | str,
        target_class: int,
    ) -> pd.DataFrame:
        """
        Суммирует частоты слов по всем XLSX-файлам для заданного класса.

        Ожидаемый формат имени файла:
            <издатель>_<серия>_<тематика>_<класс>_<слов>.xlsx

        Args:
            root_folder: корневая папка для рекурсивного обхода
            target_class: номер класса (4-й сегмент имени файла)

        Returns:
            DataFrame с колонками ['Слово', 'Сумма слов']
        """
        self.word_sums = {}
        class_re = re.compile(r'^[^_]+_[^_]+_[^_]+_(\d+)_')

        for xlsx_path in Path(root_folder).rglob('*.xlsx'):
            match = class_re.search(xlsx_path.name)
            if match and int(match.group(1)) == target_class:
                self._process_excel_file(xlsx_path)

        return pd.DataFrame(
            list(self.word_sums.items()),
            columns=['Слово', 'Сумма слов'],
        ).sort_values('Сумма слов', ascending=False)

    def _process_excel_file(self, file_path: Path) -> None:
        """Добавляет частоты из одного XLSX в общий словарь."""
        try:
            df = pd.read_excel(file_path, usecols=['Слово', 'Сумма слов'])
            for _, row in df.iterrows():
                word  = row['Слово']
                count = row['Сумма слов']
                if pd.notna(word) and pd.notna(count):
                    self.word_sums[word] = self.word_sums.get(word, 0) + int(count)
        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")

    # ------------------------------------------------------------------
    # Лексическая разница
    # ------------------------------------------------------------------
    def calculate_lexical_difference(
        self,
        file_path1: Path | str,
        file_path2: Path | str,
        output_folder: Optional[Path | str] = None,
    ) -> pd.DataFrame:
        """
        Находит слова, присутствующие в file_path2, но отсутствующие в file_path1.
        Применимо для: нахождения новой лексики следующего класса/уровня.

        Args:
            file_path1: базовый список (например, лексика 9-го класса)
            file_path2: расширенный список (например, лексика 10-го класса)
            output_folder: куда сохранить результат.
                           По умолчанию — в папку file_path1.

        Returns:
            DataFrame со словами, уникальными для file_path2.
        """
        df1 = pd.read_excel(file_path1)
        df2 = pd.read_excel(file_path2)

        words1    = set(df1['Слово'].dropna().astype(str))
        words2    = set(df2['Слово'].dropna().astype(str))
        new_words = words2 - words1

        result_df = df2[df2['Слово'].isin(new_words)].copy()

        output_dir = (
            Path(output_folder) if output_folder
            else Path(file_path1).parent
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / "lexical_difference.xlsx"
        result_df.to_excel(out_path, index=False)
        print(f"Лексическая разница сохранена: {out_path}")
        print(f"Новых слов: {len(new_words)}")

        return result_df

    # ------------------------------------------------------------------
    # Коэффициент Жуайна
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_juain_coefficient(
        result_df: pd.DataFrame,
        thematic_segments: List[str],
        col_names: List[str],
        output_column_name: str,
    ) -> pd.DataFrame:
        """
        Вычисляет коэффициент Жуайна — меру равномерности распределения
        слова по тематическим сегментам.

        Интерпретация:
            100  → слово встречается абсолютно равномерно во всех сегментах
            0    → слово сконцентрировано в одном сегменте
            -1   → недостаточно данных для вычисления
            -2   → слово отсутствует во всех сегментах (μ = 0)

        Формула:
            J = 100 * (1 - σ / (μ * √(n-1)))

        Args:
            result_df: DataFrame со словами и частотами
            thematic_segments: список названий сегментов (для логирования)
            col_names: список колонок с нормализованными частотами по сегментам
            output_column_name: название новой колонки с коэффициентом

        Returns:
            DataFrame с добавленной колонкой output_column_name
        """
        result_df = result_df.copy()
        result_df[output_column_name] = -1.0

        unique_words    = result_df['Слово'].unique()
        word_index_map  = {word: i for i, word in enumerate(unique_words)}
        n_segments      = len(col_names)
        freq_matrix     = np.zeros((len(unique_words), n_segments))

        # Заполняем матрицу частот
        for word in unique_words:
            idx       = word_index_map[word]
            word_rows = result_df[result_df['Слово'] == word].index
            for j, col in enumerate(col_names):
                if col in result_df.columns:
                    freq_matrix[idx, j] = (
                        result_df.loc[word_rows, col].fillna(0).mean()
                    )

        # Вычисляем коэффициент
        for word in tqdm(unique_words, desc=f"Жуайн: {output_column_name}"):
            idx        = word_index_map[word]
            freqs      = freq_matrix[idx]
            word_rows  = result_df[result_df['Слово'] == word].index

            if len(freqs) <= 1:
                coef = -1.0
            else:
                mu, sigma = np.mean(freqs), np.std(freqs)
                if mu > 0:
                    coef = 100.0 * (1.0 - sigma / (mu * np.sqrt(len(freqs) - 1)))
                else:
                    coef = -2.0

            result_df.loc[word_rows, output_column_name] = round(coef, 4)

        return result_df