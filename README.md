# Russian Lexicon Analyzer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NLP-инструментарий для автоматического анализа частотности русскоязычных текстов
и выявления незнакомой лексики в учебных материалах.

---

## Описание

Проект решает задачу автоматического определения слов, отсутствующих в заданном
лексиконе, применительно к учебным текстам, научной литературе и корпусам документов.

Разработан в контексте корпусной лингвистики и методики преподавания русского языка:
основная задача — помочь исследователю быстро найти лексику, выходящую за пределы
заданного словарного минимума (например, лексического ядра определённого класса).

### Основные возможности

- Подсветка незнакомых слов в DOCX-документах:
  красный — слово вне лексикона, чёрный — слово из стоп-листа
- Частотный анализ с нормализацией (ipm — instances per million words)
- Лемматизация с коррекцией ошибок через natasha и pymorphy3
- Снятие приставок при поиске в лексиконе (небыстрый -> быстрый)
- Корректная обработка дефисных конструкций (кросс-платформенный, социально-экономический)
- Распознавание и пропуск римских цифр (XIX, XIV-й)
- NER для собственных имён (Москва, Пушкин — не подсвечиваются)
- Коэффициент Жуайна — мера равномерности распределения слова по тематическим сегментам
- Пакетная обработка: рекурсивный обход папок с сохранением структуры директорий

---

## Установка

```bash
# Клонируем репозиторий
git clone https://github.com/ваш-username/russian-lexicon-analyzer.git
cd russian-lexicon-analyzer

# Создаём виртуальное окружение
python -m venv venv

# Активируем (Windows)
venv\Scripts\activate

# Активируем (Linux / Mac)
# source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt

# Скачиваем ресурсы NLTK (один раз)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Использование

### Подсветка незнакомых слов в DOCX

Основной сценарий: проверить учебный текст на соответствие лексическому минимуму.

**Минимальный запуск:**

```bash
python run_highlight.py --docx "path/to/textbook.docx" --lexicon "path/to/lexicon.xlsx"
```

**С указанием выходного файла:**

```bash
python run_highlight.py --docx "path/to/textbook.docx" --lexicon "path/to/lexicon.xlsx" --out "path/to/result.docx"
```

**С чёрным списком (слова — чёрный цвет):**

```bash
python run_highlight.py --docx "path/to/textbook.docx" --lexicon "path/to/lexicon.xlsx" --black "data/blacklist.txt"
```

**С указанием названия колонки лексикона:**

```bash
python run_highlight.py --docx "path/to/textbook.docx" --lexicon "path/to/lexicon.xlsx" --column "Слово"
```

**Сменить цвет подсветки (по умолчанию FF0000 — красный):**

```bash
python run_highlight.py --docx "path/to/textbook.docx" --lexicon "path/to/lexicon.xlsx" --color "0000FF"
```

Результат сохраняется рядом с исходным файлом с суффиксом `_highlighted.docx`.

Использование как библиотеки
Частотный анализ корпуса

from src.preprocessing.text_preprocessor import StandardTextPreprocessor
from src.analysis import FrequencyAnalyzer, FileProcessor

preprocessor = StandardTextPreprocessor()
analyzer     = FrequencyAnalyzer()
processor    = FileProcessor(preprocessor, analyzer)

# Обработка папки с DOCX-файлами
# Результат: XLSX-файлы с частотными списками
processor.create_frequency_lists(
    folder_path="data/textbooks/",
    skip_proper_nouns=True,
    output_folder="data/output/",
)

Каждый выходной XLSX содержит:

Слово	Сумма слов	Нормализованная частота
клетка	345	2300.0
организм	234	1560.0
белок	189	1260.0

Рекурсивная обработка вложенных папок

processor.create_frequency_lists_recursively(
    root_folder="data/corpora/",
    output_folder="data/output/",
    skip_proper_nouns=True,
)

Агрегация по тематическим сегментам

from src.analysis import FrequencyAggregator

aggregator = FrequencyAggregator()
aggregator.aggregate_and_normalize_by_segment(
    folder_path="data/frequencies/",
    output_folder="data/output/",
    include_book_count=True,
    include_book_percentage=True,
)

Результат — combined_frequency_list.xlsx с колонками:

Сумма слов Биология 10, Сумма слов Физика 10, ...
Нормализованная частотность Биология 10, ...
Количество учебников Биология 10
Процент учебников Биология

Лексическая разница между уровнями

from src.analysis import DataAnalyzer

analyzer  = DataAnalyzer()
new_words = analyzer.calculate_lexical_difference(
    file_path1="data/lexicon_grade_9.xlsx",
    file_path2="data/lexicon_grade_10.xlsx",
    output_folder="data/output/",
)

print(f"Новых слов в 10 классе: {len(new_words)}")

Коэффициент Жуайна
Измеряет равномерность распределения слова по тематическим сегментам.
Значение 100 означает абсолютно равномерное распределение,
0 — слово сосредоточено в одном сегменте.

from src.analysis import DataAnalyzer

result_df = DataAnalyzer.calculate_juain_coefficient(
    result_df=df,
    thematic_segments=["Биология", "Физика", "Химия"],
    col_names=[
        "Нормализованная частотность Биология 10",
        "Нормализованная частотность Физика 10",
        "Нормализованная частотность Химия 10",
    ],
    output_column_name="Коэффициент Жуайна",
)

Подсветка через API

from src.highlighting.lexicon import LexiconRepository
from src.highlighting.highlighter import DocxHighlighter
from src.preprocessing.text_preprocessor import StandardTextPreprocessor

preprocessor   = StandardTextPreprocessor()
lexicon_lemmas = LexiconRepository().load_lemmas("data/lexicon.xlsx")

highlighter = DocxHighlighter(
    preprocessor,
    lexicon_lemmas,
    unknown_hex="FF0000",
    black_hex="000000",
)
highlighter.highlight_file("input.docx", "output.docx")

Архитектура

russian-lexicon-analyzer/
│
├── src/
│   ├── config.py                     # Центральный конфиг путей
│   │
│   ├── preprocessing/
│   │   ├── utils.py                  # Вспомогательные функции
│   │   └── text_preprocessor.py     # Лемматизация, нормализация текста
│   │
│   ├── analysis/
│   │   ├── frequency_analyzer.py    # Подсчёт частот (ipm)
│   │   ├── file_processor.py        # Обработка DOCX-файлов
│   │   ├── frequency_aggregator.py  # Агрегация по сегментам
│   │   └── data_analyzer.py         # Разница лексиконов, коэфф. Жуайна
│   │
│   └── highlighting/
│       ├── lexicon.py               # Загрузка и нормализация лексикона
│       ├── matcher.py               # Классификация токенов
│       └── highlighter.py           # Подсветка в DOCX
│
├── data/
│   ├── russian_stopwords.txt        # Дополнительные стоп-слова
│   └── replacements.txt             # Словарь замен лемм
│
├── tests/
│   ├── conftest.py                  # Общие фикстуры
│   ├── test_utils.py
│   ├── test_lexicon.py
│   ├── test_frequency_analyzer.py
│   ├── test_matcher.py
│   └── test_data_analyzer.py
│
├── run_highlight.py                  # CLI-скрипт
├── requirements.txt
├── pytest.ini
└── README.md

Ключевые решения
Синглтон для NLP-моделей.
Natasha и pymorphy3 загружаются один раз и переиспользуются
всеми экземплярами препроцессора. Это критично, поскольку
загрузка моделей занимает 30-60 секунд и требует значительного объёма памяти.

Двухуровневая лемматизация.
Natasha даёт контекстно-зависимую лемматизацию, pymorphy3 — исправляет
ошибки на кратких прилагательных, причастиях и глаголах совершенного вида.
Если лемма от natasha не попадает в лексикон, но вариант pymorphy3 — попадает,
приоритет отдаётся pymorphy3.

Кэширование классификации.
Каждый токен классифицируется один раз, результат кэшируется.
В длинных документах одно и то же слово встречается многократно —
кэш даёт существенный прирост скорости.

Контекстный NER.
Распознавание именованных сущностей (PER, ORG, LOC) выполняется
на уровне всего абзаца, а не отдельных токенов. Это повышает точность:
например, «Байкал» в предложении «озеро Байкал» надёжнее распознаётся
как топоним, чем изолированно.

Тестирование

# Все быстрые тесты (без загрузки NLP-моделей)
pytest

# С отчётом о покрытии
pytest --cov=src --cov-report=html

# Только интеграционные тесты (загружают реальные модели, медленно)
pytest -m slow

# Конкретный модуль
pytest tests/test_matcher.py -v

Тесты разделены на два уровня:

Unit-тесты — используют моки NLP-моделей, выполняются за секунды
Интеграционные тесты (-m slow) — используют реальные модели,
требуют наличия файлов в data/

Технологический стек
Библиотека	Версия	Назначение
natasha	1.6.0+	Сегментация, морфология, NER для русского языка
pymorphy3	1.0.0+	Морфологический анализ, коррекция лемм
nltk	3.8.0+	Токенизация, базовые стоп-слова
python-docx	0.8.11+	Чтение и запись DOCX-документов
pandas	2.0.0+	Обработка табличных данных
numpy	1.24.0+	Матричные операции (коэффициент Жуайна)
openpyxl	3.1.0+	Чтение XLSX
xlsxwriter	3.1.0+	Запись больших XLSX (поддержка zip64)
tqdm	4.65.0+	Прогресс-бары при пакетной обработке
Производительность
Задача	Время	Объём
Подсветка одного DOCX	2-10 сек	10 000 слов
Частотный анализ	15-30 сек	100 000 слов
Агрегация 50 файлов	2-5 мин	5 000 000 слов
Тестировалось на Intel Core i5-10400, 16 GB RAM, SSD.

Основное время занимает инициализация NLP-моделей (30-60 сек, однократно).
После загрузки скорость обработки — порядка 500-1000 слов в секунду.

Известные ограничения
Латинские термины без соответствия в лексиконе всегда помечаются как unknown.
Если латинская лексика должна игнорироваться — добавьте её в стоп-лист.
NER иногда пропускает редкие имена собственные или нестандартные написания.
Дефисные конструкции из трёх и более частей обрабатываются по частям,
что в редких случаях даёт неверный результат.
Требуется Python 3.8 и выше.
