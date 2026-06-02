# src/preprocessing/text_preprocessor.py
from __future__ import annotations

import re
import string
import unicodedata
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

import pymorphy3
import nltk
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords

from src.config import DEFAULT_STOPWORDS_PATH, DEFAULT_REPLACEMENTS_PATH, resolve_path
from src.preprocessing.utils import Utils

# Один раз скачиваем нужные ресурсы NLTK (безопасно вызывать повторно)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# ---------------------------------------------------------------------------
# Синглтон для тяжёлых NLP-моделей
# Инициализируются один раз, переиспользуются всеми экземплярами препроцессора
# ---------------------------------------------------------------------------
class _NLPModels:
    """
    Хранит тяжёлые NLP-модели (natasha, pymorphy3).
    Используйте NLPModels.get() — модели загрузятся только при первом обращении.
    """
    _instance: Optional[_NLPModels] = None

    def __init__(self) -> None:
        self.segmenter   = Segmenter()
        self.emb         = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.morph_vocab  = MorphVocab()
        self.morph        = pymorphy3.MorphAnalyzer()

    @classmethod
    def get(cls) -> _NLPModels:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ---------------------------------------------------------------------------
# Базовый класс
# ---------------------------------------------------------------------------
class TextPreprocessor:
    """
    Базовый класс предобработки русскоязычного текста.

    Отвечает за:
    - загрузку стоп-слов и словаря замен из файлов
    - нормализацию текста (акценты, замена букв, спецсимволы)
    - лемматизацию через natasha + коррекцию через pymorphy3

    Подклассы обязаны реализовать filter_lemmas().
    """

    # Символы для замены (латиница → кириллица, типографские замены)
    LETTER_REPLACEMENTS: dict[str, str] = {
        'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
        'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
        'X': 'Х', 'Y': 'У', 'a': 'а', 'b': 'в', 'c': 'с',
        'e': 'е', 'h': 'н', 'k': 'к', 'm': 'м', 'o': 'о',
        'p': 'р', 't': 'т', 'x': 'х', 'y': 'у',
        'ё': 'е', 'Ё': 'Е', r'\b3': 'З', '0': 'О', '6': 'б',
    }

    # Спецсимволы для полной очистки текста
    SPEC_CHARS: str = (
        string.punctuation
        + '।\n\xa0«»\t…•1234567890°ρ‒–————­———ρ''"£‹£∆∆∠∠„№£→↔"♦✓abc′″€±'
        + '।∙∙‚"÷≤⊃⊥⋂⋃■↓∙∙∙∙≠×××∈∉∥∩∪≈⊂□←§¬®éeöμχ±νκυβοςζαοβζαβοδυναστημιςαχβχοχżženüóōáóēžžíáłżöé·®'
    )

    # Спецсимволы для мягкой очистки (для DOCX — без пунктуации)
    SPEC_CHARS_DOCX: str = (
        '…•°''"£‹£∆∆∠∠„№£→↔"♦′″।∙∙⊥⋂⋃■↓∙∙∙∙≠×××∈∉∥∩∪⊂□←¬'
        + 'éeöμχ±νκυβοςζαοβζαβοδυναστημιςαχβχοχżženüóōáóēžžíáłżöé·'
    )

    # Двухбуквенные слова, которые не нужно выбрасывать
    IMPORTANT_2LETTER: frozenset[str] = frozenset({
        'во', 'за', 'из', 'ко', 'на', 'об', 'от', 'по', 'со', 'уж',
        'мы', 'ты', 'вы', 'он', 'ей', 'их', 'бы', 'да', 'же', 'ил',
        'но', 'ну', 'то', 'ай', 'ах', 'им', 'ли', 'ми', 'ой', 'ре',
        'си', 'ту', 'яд',
    })

    def __init__(
        self,
        stopwords_path:    Optional[Path | str] = None,
        replacements_path: Optional[Path | str] = None,
    ) -> None:
        """
        Args:
            stopwords_path: путь к файлу с кастомными стоп-словами (одно слово на строку).
                            По умолчанию: data/russian_stopwords.txt
            replacements_path: путь к файлу со словарём замен (формат "ключ: значение").
                               По умолчанию: data/replacements.txt
        """
        # --- стоп-слова ---
        sw_path = resolve_path(stopwords_path, DEFAULT_STOPWORDS_PATH)
        nltk_stopwords = stopwords.words("russian")
        custom_stopwords = sw_path.read_text(encoding='utf-8').splitlines()
        custom_stopwords = [w.strip() for w in custom_stopwords if w.strip()]
        self.russian_stopwords: frozenset[str] = frozenset(nltk_stopwords + custom_stopwords)

        # --- словарь замен ---
        rp_path = resolve_path(replacements_path, DEFAULT_REPLACEMENTS_PATH)
        self.replacements: dict[str, str] = {}
        for line in rp_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                self.replacements[key.strip()] = value.strip()

        # --- NLP модели (синглтон) ---
        _models = _NLPModels.get()
        self.segmenter    = _models.segmenter
        self.emb          = _models.emb
        self.morph_tagger = _models.morph_tagger
        self.morph_vocab  = _models.morph_vocab
        self.morph        = _models.morph

    # ------------------------------------------------------------------
    # Нормализация текста
    # ------------------------------------------------------------------
    def remove_accents(self, text: str) -> str:
        """
        Удаляет диакритические знаки (акценты) из текста.
        Краткость буквы 'й' (U+0306) сохраняется намеренно.
        """
        normalized = unicodedata.normalize('NFD', text)
        return unicodedata.normalize('NFC', ''.join(
            ch for ch in normalized
            if unicodedata.category(ch) != 'Mn' or ch == '\u0306'
        ))

    def replace_letters(self, text: str) -> str:
        """
        Заменяет визуально похожие латинские буквы на кириллицу
        и исправляет типографские замены согласно LETTER_REPLACEMENTS.
        """
        for pattern, replacement in self.LETTER_REPLACEMENTS.items():
            text = re.sub(pattern, replacement, text)
        return text

    def remove_special_chars(self, text: str) -> str:
        """Полная очистка: убирает пунктуацию, цифры и спецсимволы."""
        return Utils.remove_chars_from_text(
            text, self.SPEC_CHARS + string.digits
        )

    def remove_special_chars_docx(self, text: str) -> str:
        """
        Мягкая очистка для DOCX: убирает только спецсимволы,
        оставляет пунктуацию и цифры (нужны для структуры документа).
        """
        return Utils.remove_chars_from_text(text, self.SPEC_CHARS_DOCX)

    # ------------------------------------------------------------------
    # Морфологический анализ
    # ------------------------------------------------------------------
    def _convert_short_to_full(self, parsed_form) -> str:
        """
        Преобразует краткую форму прилагательного/причастия в полную.
        Например: 'красив' → 'красивый', 'прочитан' → 'прочитанный'
        """
        for parse in self.morph.parse(parsed_form.word):
            is_full_adj  = 'ADJF' in parse.tag and 'ADJS' not in parse.tag
            is_full_part = 'PRTF' in parse.tag and 'PRTS' not in parse.tag
            if is_full_adj or is_full_part:
                return parse.normal_form
        return parsed_form.normal_form

    def _process_parsed_form(self, parsed_form) -> str:
        """Выбирает правильную лемму для разобранной формы слова."""
        if 'ADJS' in parsed_form.tag or 'PRTS' in parsed_form.tag:
            return self._convert_short_to_full(parsed_form)
        if 'VERB' in parsed_form.tag:
            for parse in self.morph.parse(parsed_form.word):
                if 'VERB' in parse.tag and 'impf' in parse.tag:
                    return parse.normal_form
        return parsed_form.normal_form

    def correct_lemma(
        self,
        token_text: str,
        natasha_lemma: str,
        existing_lemmas: List[str],
    ) -> str:
        """
        Корректирует лемму, полученную от natasha, с помощью pymorphy3.

        Natasha иногда ошибается на:
        - кратких прилагательных/причастиях
        - глаголах совершенного вида

        Returns:
            Исправленная лемма с заменой ё→е
        """
        def _replace_yo(s: str) -> str:
            return s.replace('ё', 'е').replace('Ё', 'Е')

        parsed = self.morph.parse(natasha_lemma)
        if not parsed or natasha_lemma.lower() == token_text.lower():
            parsed_token = self.morph.parse(token_text)
            if parsed_token:
                return _replace_yo(self._process_parsed_form(parsed_token[0]))
            return _replace_yo(token_text)

        best_parse = parsed[0]

        if 'ADJS' in best_parse.tag or 'PRTS' in best_parse.tag:
            return _replace_yo(self._convert_short_to_full(best_parse))

        if 'VERB' in best_parse.tag:
            for parse in parsed:
                if 'VERB' in parse.tag and 'impf' in parse.tag:
                    return _replace_yo(parse.normal_form)
            return _replace_yo(best_parse.normal_form)

        return _replace_yo(natasha_lemma)

    # ------------------------------------------------------------------
    # Лемматизация — переопределяется в подклассах
    # ------------------------------------------------------------------
    @abstractmethod
    def lemmatize(self, text: str, **kwargs) -> List[str]:
        """Лемматизирует текст и возвращает список лемм."""
        raise NotImplementedError

    @abstractmethod
    def filter_lemmas(self, lemmas: List[str]) -> List[str]:
        """Фильтрует леммы после лемматизации."""
        raise NotImplementedError

    def preprocess(self, text: str, **kwargs) -> List[str]:
        """
        Полный пайплайн предобработки:
        1. Удаление акцентов
        2. Замена похожих букв
        3. Удаление спецсимволов
        4. Лемматизация
        5. Фильтрация

        Args:
            text: исходный текст
            **kwargs: передаются в lemmatize() (например, skip_proper_nouns=True)

        Returns:
            Список очищенных лемм
        """
        text = self.remove_accents(text)
        text = self.replace_letters(text)
        text = self.remove_special_chars(text)
        lemmas = self.lemmatize(text, **kwargs)
        return self.filter_lemmas(lemmas)


# ---------------------------------------------------------------------------
# Подкласс 1: стандартная предобработка
# ---------------------------------------------------------------------------
class StandardTextPreprocessor(TextPreprocessor):
    """
    Стандартная предобработка текста.

    Отличия от базового класса:
    - Собственные имена (Geox, Name, Surn) опционально пропускаются
    - Фильтрация удаляет месяцы, латиницу, одиночные буквы,
      двухбуквенные слова и стоп-слова
    """

    # Паттерн для месяцев — чтобы не попадали в частотный список
    _MONTH_RE = re.compile(
        r'\b(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]'
        r'|июн[ья]|июл[ья]|август[а]?|сентябр[ья]'
        r'|октябр[ья]|ноябр[ья]|декабр[ья])\b',
        flags=re.IGNORECASE,
    )

    def lemmatize(
        self,
        text: str,
        skip_proper_nouns: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Лемматизирует текст с помощью natasha + коррекция pymorphy3.

        Args:
            text: очищенный текст
            skip_proper_nouns: если True — собственные имена пропускаются

        Returns:
            Список лемм (строки в нижнем регистре, ё→е)
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        lemmas: List[str] = []
        _SKIP_POS = {'PRON', 'NUM', 'ADP', 'SCONJ', 'CCONJ', 'PART', 'INTJ', 'PUNCT', 'SYM'}
        _PROPER_TAGS = ('Geox', 'Name', 'Surn')

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            pos   = token.pos
            lemma = (token.lemma or '').replace('ё', 'е').replace('Ё', 'Е')

            corrected = self.correct_lemma(token.text, lemma, lemmas)
            parsed    = self.morph.parse(corrected)
            if not parsed:
                continue

            best = parsed[0]

            # Выбрасываем служебные части речи и слова с нулевым скором
            if best.score <= 0 or pos in _SKIP_POS:
                continue

            is_proper = any(tag in best.tag for tag in _PROPER_TAGS)
            if is_proper and skip_proper_nouns:
                continue

            # Собственные имена — с заглавной, остальные — в нижнем регистре
            final_lemma = corrected.title() if is_proper else corrected.lower()
            final_lemma = final_lemma.replace('ё', 'е').replace('Ё', 'Е')
            lemmas.append(self.replacements.get(final_lemma, final_lemma))

        return lemmas

    def filter_lemmas(self, lemmas: List[str]) -> List[str]:
        """
        Фильтрует список лемм:
        - убирает названия месяцев
        - убирает латинские слова и римские цифры
        - убирает слова длиной 1-2 символа
        - убирает стоп-слова
        """
        text = ' '.join(lemmas)
        text = re.sub(r'\b[a-zA-Z]+\b',    ' ', text)   # латиница
        text = re.sub(r'\b[ivxlcdm]+\b',   ' ', text)   # римские цифры
        text = self._MONTH_RE.sub(' ', text)
        text = re.sub(r'\b\w\b',            ' ', text)   # однобуквенные
        text = re.sub(r'\b\w{2}\b|\b\w-\w\b', ' ', text)  # двухбуквенные
        text = re.sub(r'\b3', 'З', text)

        tokens = word_tokenize(text)
        return [t.strip() for t in tokens if t not in self.russian_stopwords]


# ---------------------------------------------------------------------------
# Подкласс 2: предобработка без стоп-слов (для анализа сочетаемости)
# ---------------------------------------------------------------------------
class TextPreprocessorWithoutStopwords(TextPreprocessor):
    """
    Предобработка без фильтрации стоп-слов.
    Используется когда важно сохранить служебные слова
    (например, при анализе n-грамм или сочетаемости).

    Отличия:
    - Собственные имена всегда пропускаются
    - Двухбуквенные слова сохраняются, если они в IMPORTANT_2LETTER
    - Стоп-слова НЕ фильтруются на последнем шаге
    """

    _MONTH_RE = re.compile(
        r'\b(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]'
        r'|июн[ья]|июл[ья]|август[а]?|сентябр[ья]'
        r'|октябр[ья]|ноябр[ья]|декабр[ья])\b',
        flags=re.IGNORECASE,
    )

    def lemmatize(self, text: str, **kwargs) -> List[str]:
        """
        Лемматизирует текст, исключая собственные имена и служебные POS.
        Стоп-слова на этом этапе НЕ фильтруются.
        """
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        lemmas: List[str] = []
        _SKIP_POS    = {'PRON', 'NUM', 'ADP', 'SCONJ', 'CCONJ', 'PART', 'INTJ', 'PUNCT', 'SYM'}
        _PROPER_TAGS = ('Geox', 'Name', 'Surn')

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            pos = token.pos

            corrected = self.correct_lemma(token.text, token.lemma or '', lemmas)
            parsed    = self.morph.parse(corrected)
            if not parsed:
                continue

            best = parsed[0]

            skip_conditions = (
                best.score <= 0
                or pos in _SKIP_POS
                or any(tag in best.tag for tag in _PROPER_TAGS)
            )
            if skip_conditions:
                continue

            lemma = corrected.lower().replace('ё', 'е').replace('Ё', 'Е')
            lemmas.append(self.replacements.get(lemma, lemma))

        return lemmas

    def filter_lemmas(self, lemmas: List[str]) -> List[str]:
        """
        Фильтрует список лемм:
        - убирает латиницу, римские цифры, названия месяцев
        - сохраняет двухбуквенные слова из IMPORTANT_2LETTER
        """
        text = ' '.join(lemmas)
        text = re.sub(r'\b[a-zA-Z]+\b',  ' ', text)
        text = re.sub(r'\b[ivxlcdm]+\b', ' ', text)
        text = self._MONTH_RE.sub(' ', text)

        # Однобуквенные (кроме гласных-союзов а, и, о, у, я, с, к, в)
        text = re.sub(r'\b[^явксуоаи]\b', ' ', text)

        # Двухбуквенные — оставляем только из важного списка
        important_pattern = '|'.join(
            re.escape(w) for w in sorted(self.IMPORTANT_2LETTER, key=len, reverse=True)
        )
        two_letter_re = re.compile(
            rf'(?<!\S)(?!(?:{important_pattern})\b)[а-яё]{{2}}(?!\S)'
        )
        text = two_letter_re.sub(' ', text)

        return word_tokenize(text)