# src/analysis/frequency_aggregator.py
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


class FrequencyAggregator:
    """
    Агрегирует частотные списки из нескольких XLSX-файлов.

    Соглашение по именованию файлов (для сегментации):
        <издатель>_<серия>_<тематика>_<класс>_<слов>.xlsx
        Пример: drofa_rainbow_history_10_175000.xlsx
    """

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_total_words_from_filename(filename: str) -> int:
        """Извлекает кол-во слов из имени файла (последнее число перед .xlsx)."""
        match = re.search(r'_(\d+)\.xlsx$', filename, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _extract_segment_key(filename: str) -> tuple[str, str]:
        """
        Извлекает (тематика, класс) из имени файла.
        Формат: <...>_<тематика>_<класс>_<слов>.xlsx
        """
        parts = Path(filename).stem.split('_')
        thematic  = parts[2] if len(parts) >= 4 else 'Unknown'
        class_num = parts[3] if len(parts) >= 4 else 'Unknown'
        return thematic, class_num

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------
    def aggregate_and_normalize_by_segment(
        self,
        folder_path: Path | str,
        output_folder: Path | str,
        include_doc_counts:     bool = True,
        include_book_percentage: bool = True,
        include_book_count:     bool = True,
    ) -> None:
        """
        Создаёт сводный частотный список с разбивкой по тематическим сегментам.

        Для каждого слова вычисляет:
        - абсолютную и нормализованную частоту по каждому сегменту (тематика + класс)
        - кол-во учебников, в которых встречается слово
        - процент учебников в тематике

        Args:
            folder_path: папка с XLSX частотными списками
            output_folder: куда сохранять результат
            include_doc_counts: добавить колонки с частотой по каждому файлу
            include_book_percentage: добавить процент учебников в тематике
            include_book_count: добавить кол-во учебников на сегмент
        """
        folder_path   = Path(folder_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        if not files:
            print("XLSX-файлы не найдены.")
            return

        # --- загрузка ---
        print("Загрузка файлов...")
        file_data: dict[str, pd.DataFrame] = {
            f: pd.read_excel(folder_path / f, usecols=[0, 1])
            for f in tqdm(files, desc="Чтение")
        }

        # --- сбор метаданных ---
        metadata:         dict = {}          # key → {total_words, word_counts, files}
        thematic_counts:  dict = defaultdict(int)
        total_word_counts: dict = defaultdict(int)

        for file, df in file_data.items():
            thematic, class_num = self._extract_segment_key(file)
            key         = f"{thematic} {class_num}"
            total_words = self._extract_total_words_from_filename(file)

            if key not in metadata:
                metadata[key] = {
                    'total_words': 0,
                    'word_counts': defaultdict(int),
                    'files':       [],
                }

            metadata[key]['total_words'] += total_words
            metadata[key]['files'].append(file)
            thematic_counts[thematic] += total_words

            for _, row in df.iterrows():
                word  = row.iloc[0]
                count = row.iloc[1]
                if pd.notna(word) and pd.notna(count):
                    metadata[key]['word_counts'][word] += int(count)
                    total_word_counts[word] += int(count)

        # --- построение результата ---
        result = []
        for word in tqdm(total_word_counts, desc="Построение таблицы"):
            row_data = {
                'Слово':    word,
                'Количество': total_word_counts[word],
            }
            for key, data in metadata.items():
                thematic = key.split(' ')[0]
                count    = data['word_counts'].get(word, 0)
                norm_freq = (
                    round((count / data['total_words']) * 1_000_000, 2)
                    if data['total_words'] > 0 else 0
                )
                row_data[f'Сумма слов {key}']               = count
                row_data[f'Нормализованная частотность {key}'] = norm_freq

                if include_book_count:
                    books_count = sum(
                        1 for f in data['files']
                        if file_data[f][file_data[f].columns[0]].eq(word).any()
                    )
                    row_data[f'Количество учебников {key}'] = books_count

                if include_doc_counts:
                    for f in data['files']:
                        mask  = file_data[f].iloc[:, 0] == word
                        count_in_file = int(file_data[f].loc[mask].iloc[:, 1].sum())
                        row_data[f'Количество в {f}'] = count_in_file

                if include_book_percentage:
                    total_books_th = sum(
                        len(v['files'])
                        for k, v in metadata.items()
                        if k.startswith(f"{thematic} ")
                    )
                    books_with_word = sum(
                        1
                        for k, v in metadata.items()
                        if k.startswith(f"{thematic} ")
                        for f in v['files']
                        if file_data[f].iloc[:, 0].eq(word).any()
                    )
                    pct = (
                        round((books_with_word / total_books_th) * 100, 2)
                        if total_books_th > 0 else 0
                    )
                    row_data[f'Процент учебников {thematic}'] = pct

            result.append(row_data)

        # --- сохранение ---
        df_result = pd.DataFrame(result).sort_values('Количество', ascending=False)
        out_path  = output_folder / "combined_frequency_list.xlsx"

        with pd.ExcelWriter(str(out_path), engine='xlsxwriter') as writer:
            writer.book.use_zip64()
            df_result.to_excel(writer, index=False)

        print(f"Сохранено: {out_path}")

    def normalize_frequencies_across_thematic(
        self,
        directory_path: Path | str,
        output_folder:  Optional[Path | str] = None,
        name:           str = "общий",
    ) -> None:
        """
        Объединяет все XLSX из папки в один общий частотный список.
        Нормализация — на 1 миллион слов (ipm).

        Args:
            directory_path: папка с XLSX-файлами
            output_folder: куда сохранить результат (по умолчанию — туда же)
            name: суффикс в имени выходного файла
        """
        directory_path = Path(directory_path)
        output_dir     = Path(output_folder) if output_folder else directory_path
        output_dir.mkdir(parents=True, exist_ok=True)

        all_data:    list[pd.DataFrame] = []
        total_words: int = 0

        for xlsx_file in directory_path.glob('*.xlsx'):
            words = self._extract_total_words_from_filename(xlsx_file.name)
            total_words += words
            try:
                df = pd.read_excel(xlsx_file, usecols=['Слово', 'Сумма слов'])
                all_data.append(df)
            except Exception as e:
                print(f"Ошибка при чтении {xlsx_file.name}: {e}")

        if not all_data:
            print("Нет данных для обработки.")
            return

        result_df = (
            pd.concat(all_data, ignore_index=True)
            .groupby('Слово', as_index=False)['Сумма слов']
            .sum()
        )
        result_df['Нормализованная частотность'] = (
            (result_df['Сумма слов'] / total_words) * 1_000_000
            if total_words > 0 else 0
        )

        out_path = output_dir / f"Общий частотный список {name}.xlsx"
        result_df.to_excel(out_path, index=False)
        print(f"Сохранено: {out_path}")