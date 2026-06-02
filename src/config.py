# D:\russian-lexicon-analyzer\src\config.py
from pathlib import Path

# Корень проекта — вычисляется автоматически относительно этого файла
# src/config.py находится в D:\russian-lexicon-analyzer\src\
# значит .parent.parent = D:\russian-lexicon-analyzer\
PROJECT_ROOT = Path(__file__).parent.parent

# Папка с данными
DATA_DIR = PROJECT_ROOT / "data"

# Файлы по умолчанию
DEFAULT_STOPWORDS_PATH    = DATA_DIR / "russian_stopwords.txt"
DEFAULT_REPLACEMENTS_PATH = DATA_DIR / "replacements.txt"


def resolve_path(path, default: Path) -> Path:
    """
    Возвращает Path:
      - если path задан явно — проверяет существование и возвращает его
      - если path=None — возвращает default
    """
    result = Path(path) if path is not None else default
    if not result.exists():
        raise FileNotFoundError(
            f"Файл не найден: {result}\n"
            f"Укажите корректный путь или положите файл в {default}"
        )
    return result