# run_highlight.py
"""
Скрипт подсветки незнакомых слов в DOCX.

Запуск из корня проекта:
    python run_highlight.py
"""
from pathlib import Path

# --- Пути ---
DOCX_PATH = Path(
    r"C:\Users\spot2\OneDrive\Рабочий стол\Частотные списки и учебники МИ"
    r"\Пасечник\Учебники\Пасечник 11 класс 2025_61463.docx"
)
XLSX_PATH = Path(
    r"C:\Users\spot2\OneDrive\Рабочий стол\Частотные списки и учебники МИ"
    r"\Лексическое ядро\Частотные списки\CombinedLexicalCores"
    r"\10 класс\Лексическое ядро 10 класс ALL.xlsx"
)

# Выходной файл — рядом с исходным, суффикс _highlighted
OUT_PATH = DOCX_PATH.with_name(DOCX_PATH.stem + "_highlighted.docx")

# --- Импорты ---
from src.preprocessing.text_preprocessor import StandardTextPreprocessor
from src.highlighting.lexicon import LexiconRepository
from src.highlighting.highlighter import DocxHighlighter

# --- Запуск ---
def main():
    # Проверяем что файлы существуют
    if not DOCX_PATH.exists():
        raise FileNotFoundError(f"DOCX не найден:\n  {DOCX_PATH}")
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Лексикон не найден:\n  {XLSX_PATH}")

    print(f"DOCX:    {DOCX_PATH.name}")
    print(f"Лексикон: {XLSX_PATH.name}")
    print(f"Выход:   {OUT_PATH.name}")
    print()

    # Загружаем лексикон
    print("Загрузка лексикона...")
    lex_repo       = LexiconRepository(lemma_column_name=None)
    lexicon_lemmas = lex_repo.load_lemmas(XLSX_PATH)
    print(f"Лемм в лексиконе: {len(lexicon_lemmas)}")

    # Инициализируем препроцессор
    print("Инициализация препроцессора...")
    pre = StandardTextPreprocessor()

    # Запускаем подсветку
    print("Подсветка документа...")
    highlighter = DocxHighlighter(
        pre,
        lexicon_lemmas,
        unknown_hex="FF0000",   # красный — незнакомые слова
        black_hex="000000",     # чёрный  — слова из чёрного списка
    )
    highlighter.highlight_file(DOCX_PATH, OUT_PATH)

    print(f"\nГотово: {OUT_PATH}")


if __name__ == "__main__":
    main()