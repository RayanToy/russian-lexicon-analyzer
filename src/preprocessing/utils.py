# src/preprocessing/utils.py
import os
from pathlib import Path
import numpy as np
import pandas as pd


class Utils:
    @staticmethod
    def remove_chars_from_text(text: str, chars: str) -> str:
        """
        Удаляет указанные символы из текста, заменяя их пробелами.
        Используется для очистки текста от пунктуации и спецсимволов.
        """
        return "".join([' ' if ch in chars else ch for ch in text])

    @staticmethod
    def load_dataframe(file_path: str | Path) -> pd.DataFrame:
        """
        Загружает данные из файла Excel (.xlsx) или CSV (.csv).

        Args:
            file_path: путь к файлу

        Returns:
            pd.DataFrame с данными из файла

        Raises:
            ValueError: если формат файла не поддерживается
            FileNotFoundError: если файл не существует
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        ext = file_path.suffix.lower()
        if ext == '.xlsx':
            return pd.read_excel(file_path)
        elif ext == '.csv':
            return pd.read_csv(file_path)
        raise ValueError(
            f"Неподдерживаемый формат файла: {ext}. "
            f"Поддерживаются: .xlsx, .csv"
        )

    @staticmethod
    def merge_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        key_column1: str,
        key_column2: str,
    ) -> pd.DataFrame:
        """
        Объединяет два DataFrame по ключевым столбцам (left join).
        Числовые пропуски заполняются 0, строковые — пустой строкой.

        Args:
            df1: основной DataFrame
            df2: присоединяемый DataFrame
            key_column1: ключевой столбец в df1
            key_column2: ключевой столбец в df2 (будет переименован в key_column1)

        Returns:
            Объединённый DataFrame
        """
        df2 = df2.rename(columns={key_column2: key_column1})
        merged = pd.merge(df1, df2, on=key_column1, how='left')

        numeric_cols = merged.select_dtypes(include=[np.number]).columns
        merged[numeric_cols] = merged[numeric_cols].fillna(0)

        string_cols = merged.select_dtypes(include=[object]).columns
        merged[string_cols] = merged[string_cols].fillna('')

        return merged