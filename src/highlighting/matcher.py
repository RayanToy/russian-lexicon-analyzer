# src/highlighting/matcher.py
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Set

from src.highlighting.lexicon import LemmaNormalizer
from src.preprocessing.text_preprocessor import StandardTextPreprocessor

logger = logging.getLogger("lex_highlighter")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)


# ---------------------------------------------------------------------------
# Вспомогательные функции нормализации
# ---------------------------------------------------------------------------

# Символы ударения — удаляем только их, не трогая 'й' (U+0306)
_STRESS_CHARS: Set[str] = {
    "\u0301",  # COMBINING ACUTE ACCENT
    "\u0341",  # COMBINING ACUTE TONE MARK
    "\u00B4",  # SPACING ACUTE ACCENT
    "\u02CA",  # MODIFIER LETTER ACUTE ACCENT
}

# Латинские гласные с акутом → кириллица + combining acute
_LATIN_STRESS_MAP = str.maketrans({
    "á": "а\u0301", "Á": "А\u0301",
    "é": "е\u0301", "É": "Е\u0301",
    "í": "и\u0301", "Í": "И\u0301",
    "ó": "о\u0301", "Ó": "О\u0301",
    "ú": "у\u0301", "Ú": "У\u0301",
})


def strip_stress_only(s: str) -> str:
    """Удаляет знаки ударения, сохраняя краткость 'й'."""
    return ''.join(ch for ch in s if ch not in _STRESS_CHARS)


def normalize_latin_stress_vowels(s: str) -> str:
    """Заменяет латинские á/é/… на кириллицу + combining acute."""
    return s.translate(_LATIN_STRESS_MAP)


# ---------------------------------------------------------------------------
# Матчер лемм и токенов
# ---------------------------------------------------------------------------

class TokenLemmaMatcher:
    """
    Классифицирует токены текста относительно лексикона.

    Категории:
        'known'      — слово есть в лексиконе → не подсвечивать
        'unknown'    — слова нет в лексиконе  → красный
        'stop_black' — слово из чёрного списка → чёрный
        'stopword'   — служебное/местоимение/короткое → не подсвечивать

    Особые случаи, которые обрабатываются корректно:
        - Дефисные конструкции (само-управление, кросс-платформенный)
        - Слова с приставками (небыстрый → быстрый)
        - Римские цифры (XIX, XIV → stopword)
        - Собственные имена через NER (Москва, Пётр → stopword)
        - Латиница (unknown если не в лексиконе)
        - Слова с ударением (кото́в → котов)
    """

    # --- Регулярные выражения ---
    DASH_RE       = re.compile(r"[‐‑–—-]+")
    HYPH_SPLIT_RE = re.compile(r"[\u00AD‐‑–—-]+")   # split по дефисам + soft hyphen

    # Валидатор римских цифр
    ROMAN_VALID_RE = re.compile(
        r"^(?=[MDCLXVI]+$)"
        r"M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
    )

    # Суффиксы порядковых числительных: XIX-й, XIX-го и т.п.
    ORDINAL_SUFFIXES: Set[str] = {
        'й', 'я', 'е', 'го', 'му', 'м', 'ми', 'х',
        'ая', 'ое', 'ые', 'ую', 'ого', 'ому', 'ых',
        'ыми', 'им', 'ем', 'ом', 'ой',
    }

    # Расширенный список приставок (нормализованы: нижний регистр, ё→е)
    DEFAULT_STRIP_PREFIXES: Set[str] = {
        # Базовые русские
        'без', 'бес', 'в', 'во', 'воз', 'вос', 'вз', 'вс', 'взо', 'вы',
        'вне', 'внутри', 'до', 'за', 'из', 'ис', 'изо', 'меж', 'между',
        'на', 'над', 'надо', 'не', 'недо', 'небез', 'низ', 'нис',
        'о', 'об', 'обо', 'от', 'ото', 'пере', 'по', 'под', 'подо',
        'пред', 'преди', 'при', 'про', 'раз', 'рас', 'разо',
        'с', 'со', 'су', 'у', 'около', 'само',
        # Иноязычные
        'анти', 'контр', 'противо', 'де', 'дез', 'дис',
        'ин', 'им', 'ир', 'ил', 'ре',
        'супер', 'ультра', 'сверх', 'квази', 'псевдо',
        'мета', 'крипто', 'кибер', 'нейро',
        'мега', 'гига', 'тера', 'пико', 'нано', 'микро', 'макро',
        'милли', 'санти', 'деци', 'кило', 'гекто', 'дека', 'фемто', 'атто',
        'моно', 'би', 'три', 'тетра', 'пента', 'гекса', 'гепта',
        'окта', 'эннеа', 'поли',
        'кросс', 'транс', 'интер', 'интра', 'инфра',
        'экстра', 'экс', 'нео', 'ретро', 'пост', 'пра', 'пан',
        'пара', 'супра', 'гипер', 'гипо', 'выше',
        # Составные основы (агро-, авиа-, ...)
        'агро', 'авиа', 'авто', 'аэро', 'аква', 'аудио', 'видео',
        'радио', 'фото', 'электро', 'энерго', 'теле', 'опто',
        'термо', 'гидро', 'гео', 'био', 'эко', 'евро',
        'этно', 'социо', 'космо', 'кванто', 'турбо',
        # Должностные (через дефис)
        'вице', 'экс', 'обер', 'унтер',
    }

    def __init__(
        self,
        preprocessor: StandardTextPreprocessor,
        lexicon_lemmas: Set[str],
        stopword_lemmas: Optional[Set[str]] = None,
        black_stopword_lemmas: Optional[Set[str]] = None,
        strip_prefixes: Optional[Set[str]] = None,
    ) -> None:
        """
        Args:
            preprocessor: препроцессор для лемматизации токенов
            lexicon_lemmas: множество «известных» лемм
            stopword_lemmas: служебные слова (не подсвечиваем).
                             По умолчанию берётся из preprocessor.russian_stopwords
            black_stopword_lemmas: слова из чёрного списка (подсвечиваем чёрным)
            strip_prefixes: приставки для снятия при поиске в лексиконе.
                            По умолчанию — DEFAULT_STRIP_PREFIXES
        """
        self.pre     = preprocessor
        self.lexicon = {LemmaNormalizer.normalize(x) for x in (lexicon_lemmas or set())}

        # Стоп-слова: из аргумента или из препроцессора
        if stopword_lemmas is None and hasattr(preprocessor, "russian_stopwords"):
            stopword_lemmas = {
                LemmaNormalizer.normalize(w)
                for w in preprocessor.russian_stopwords
            }
        self.stopwords      = {LemmaNormalizer.normalize(x) for x in (stopword_lemmas or set())}
        self.black_stopwords = {LemmaNormalizer.normalize(x) for x in (black_stopword_lemmas or set())}

        # Приставки: сортируем по убыванию длины (жадный матч)
        prefixes_source = strip_prefixes or self.DEFAULT_STRIP_PREFIXES
        self.strip_prefixes = tuple(sorted(
            {LemmaNormalizer.normalize(p).strip('-') for p in prefixes_source if p},
            key=len, reverse=True,
        ))
        self._strip_prefixes_set = set(self.strip_prefixes)

        # Слова, которые всегда считаем «не нуждающимися в подсветке»
        self.force_green: Set[str] = {'который'}

        # Кэши для производительности
        self._lemma_cache:   dict[str, str]           = {}
        self._class_cache:   dict[str, str]           = {}
        self._parses_cache:  dict[str, list]          = {}
        self._cand_lemmas_cache: dict[str, Set[str]]  = {}

        # Debug-токены (задаются через переменную окружения LEX_DEBUG_TOKENS)
        env_dbg = os.getenv("LEX_DEBUG_TOKENS", "")
        self.debug_tokens: Set[str] = {
            LemmaNormalizer.normalize(w)
            for w in env_dbg.split(",") if w.strip()
        }

        logger.debug(
            f"[init] lexicon={len(self.lexicon)}, "
            f"stopwords={len(self.stopwords)}, "
            f"black_stopwords={len(self.black_stopwords)}"
        )

    # ------------------------------------------------------------------
    # Внутренние хелперы
    # ------------------------------------------------------------------
    def _strip_dashes(self, s: str) -> str:
        return self.DASH_RE.sub("", s)

    def _split_hyphen_parts(self, token: str) -> List[str]:
        return [p for p in self.HYPH_SPLIT_RE.split(token) if p]

    def _norm_for_parse(self, token: str) -> str:
        """Нормализует токен перед передачей в морфологический анализатор."""
        if self._is_latin(token):
            return self._strip_dashes(token.replace('\u00AD', ''))

        t = normalize_latin_stress_vowels(token)
        t = strip_stress_only(t)
        t = self.pre.replace_letters(t)
        t = t.replace('\u00AD', '')
        return self._strip_dashes(t)

    def _get_parses(self, token: str) -> list:
        """Возвращает разборы pymorphy3 с кэшированием."""
        if token not in self._parses_cache:
            t = self._norm_for_parse(token)
            self._parses_cache[token] = self.pre.morph.parse(t)
        return self._parses_cache[token]

    def _candidate_lemmas(self, token: str) -> Set[str]:
        """
        Все возможные нормальные формы токена по pymorphy3.
        Нужны для учёта омонимии (например, 'стали' → {'сталь', 'стать'}).
        """
        if token not in self._cand_lemmas_cache:
            parses = self._get_parses(token)
            cands  = {
                LemmaNormalizer.normalize(p.normal_form)
                for p in parses
                if getattr(p, 'normal_form', None)
            }
            cands.discard('')
            self._cand_lemmas_cache[token] = cands
        return self._cand_lemmas_cache[token]

    def _lemma_of(self, token: str) -> str:
        """
        Получает лемму токена:
        1. Через внешний лемматизатор (natasha + pymorphy3)
        2. Если лемма не в лексиконе, но есть вариант от pymorphy3 — берём его
        3. Применяем словарь замен

        Результат кэшируется.
        """
        if token in self._lemma_cache:
            return self._lemma_cache[token]

        t = self._norm_for_parse(token)
        if not t:
            self._lemma_cache[token] = ""
            return ""

        # 1) Лемматизация через препроцессор
        try:
            lemmas = self.pre.lemmatize(t, skip_proper_nouns=False)
        except TypeError:
            lemmas = self.pre.lemmatize(t)
        lemma = LemmaNormalizer.normalize(lemmas[0]) if lemmas else ""

        # 2) Лучший вариант pymorphy3
        parses  = self._get_parses(token)
        pym_best = (
            LemmaNormalizer.normalize(parses[0].normal_form)
            if parses else ""
        )

        # Предпочтём pymorphy3, если он даёт попадание в лексикон
        if pym_best and (lemma not in self.lexicon) and (pym_best in self.lexicon):
            lemma = pym_best

        # Фолбэк
        if not lemma:
            lemma = pym_best or LemmaNormalizer.normalize(t)

        # 3) Доменные замены
        if hasattr(self.pre, "replacements") and self.pre.replacements:
            lemma = self.pre.replacements.get(lemma, lemma)

        self._lemma_cache[token] = lemma

        if self._dbg_on(token, lemma):
            logger.debug(f"[lemma] token='{token}' -> lemma='{lemma}'")

        return lemma

    def _lemma_of_part(self, part: str) -> str:
        return self._lemma_of(part) if part else ""

    # ------------------------------------------------------------------
    # Флаги категорий токена
    # ------------------------------------------------------------------
    def _is_latin(self, token: str) -> bool:
        return bool(re.fullmatch(r'[A-Za-z]+', self._strip_dashes(token)))

    def _is_short(self, token: str) -> bool:
        return len(self._strip_dashes(token)) < 3

    def _is_roman(self, token: str) -> bool:
        t = LemmaNormalizer.normalize(self._strip_dashes(token)).upper()
        return bool(self.ROMAN_VALID_RE.match(t)) if t else False

    def _is_pronoun(self, token: str) -> bool:
        return any(
            ('NPRO' in p.tag) or ('Apro' in p.tag)
            for p in self._get_parses(token)
        )

    def _is_function_word(self, token: str) -> bool:
        return any(
            ('PREP' in p.tag) or ('CONJ' in p.tag) or ('PRCL' in p.tag)
            for p in self._get_parses(token)
        )

    def _is_force_green(self, token: str, lemma: Optional[str] = None) -> bool:
        tnorm = LemmaNormalizer.normalize(token)
        return tnorm in self.force_green or bool(lemma and lemma in self.force_green)

    def _is_proper_strict(self, token: str) -> bool:
        """
        Определяет, является ли токен именем собственным.

        Стратегия (по убыванию надёжности):
        1. NER через natasha (с кэшированием контекста абзаца)
        2. Морфологический анализ (теги Name, Surn, Patr, Geox, Orgn)
        3. Эвристика для гео-прилагательных (Галапагосских, Байкальский)
        """
        tok = (token or '').strip()
        if not tok:
            return False

        # Латиницу через NER не распознаём (не кириллица)
        if re.fullmatch(r'[A-Za-z]+', self._strip_dashes(tok)):
            return False

        # --- 1. NER ---
        try:
            if not hasattr(self, '_ner_ready'):
                from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
                self._ner_segmenter = Segmenter()
                self._ner_tagger    = NewsNERTagger(NewsEmbedding())
                self._ner_Doc       = Doc
                self._ner_ready     = True

            context = getattr(self, '_ner_context', None)
            DocCls  = getattr(self, '_ner_Doc', None)

            if DocCls and isinstance(context, str) and tok in context:
                # Кэшируем NER для всего абзаца целиком
                if getattr(self, '_ner_ctx_cache_text', None) != context:
                    doc = DocCls(context)
                    doc.segment(self._ner_segmenter)
                    doc.tag_ner(self._ner_tagger)
                    self._ner_ctx_cache_text = context
                    self._ner_ctx_spans = [
                        (span.start, span.stop, span.type)
                        for span in doc.spans
                        if span.type in ('PER', 'ORG', 'LOC')
                    ]

                for start, stop, _ in getattr(self, '_ner_ctx_spans', []):
                    span_text = context[start:stop]
                    if re.search(
                        rf'(?<!\w){re.escape(tok)}(?!\w)',
                        span_text, flags=re.UNICODE
                    ):
                        return True

            elif DocCls:
                # Без контекста — NER на одном токене
                doc = DocCls(tok)
                doc.segment(self._ner_segmenter)
                doc.tag_ner(self._ner_tagger)
                for span in doc.spans:
                    if (span.type in ('PER', 'ORG', 'LOC')
                            and span.start == 0
                            and span.stop == len(tok)):
                        return True
        except Exception:
            pass

        # --- 2. Морфология ---
        parses = self._get_parses(tok)
        if parses:
            wanted        = ('Name', 'Surn', 'Patr', 'Geox', 'Orgn')
            proper_parses = [p for p in parses if any(w in p.tag for w in wanted)]
            if proper_parses:
                best        = max(proper_parses, key=lambda p: p.score)
                competitors = [
                    p for p in parses
                    if p is not best and ('NOUN' in p.tag or 'ADJF' in p.tag)
                ]
                if not any(p.score >= best.score - 0.1 for p in competitors):
                    return True

        # --- 3. Эвристика для гео-прилагательных ---
        is_title = bool(tok[:1].isupper()) and not tok.isupper()
        if is_title and re.search(
            r"[Сс]к(ий|ая|ое|ие|ого|ому|ым|ом|ой|ую|их|ими)$", tok
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Приставки
    # ------------------------------------------------------------------
    def _prefix_base_forms(self, token: str):
        """
        Генерирует варианты основы после снятия приставок.
        Например: 'небыстрый' → 'быстрый'
        """
        t = LemmaNormalizer.normalize(token).replace('\u00AD', '')
        t = self._strip_dashes(t)
        for pref in self.strip_prefixes:
            if t.startswith(pref) and len(t) > len(pref) + 1:
                yield t[len(pref):]

    # ------------------------------------------------------------------
    # Классификация дефисных конструкций
    # ------------------------------------------------------------------
    def classify_hyphenated(self, parts: List[str]) -> str:
        """
        Классифицирует дефисное слово по его частям.

        Логика:
        1. Всё из римских цифр → stopword
        2. XIX-й, XIX-го и т.п. → stopword
        3. Любая часть в чёрном списке → stop_black
        4. Все нерим. части — стоп-слова → stopword
        5. Первая часть — префикс, вторая в лексиконе → known
        6. Все нерим. части в лексиконе → known
        7. Иначе → unknown
        """
        part_lemmas    = [self._lemma_of_part(p) for p in parts if p]
        is_roman_flags = [self._is_roman(p) for p in parts]

        # 1. Всё — римские цифры
        if all(is_roman_flags):
            return 'stopword'

        # 2. Римская цифра + порядковый суффикс
        if (len(parts) == 2
                and is_roman_flags[0]
                and LemmaNormalizer.normalize(parts[1]) in self.ORDINAL_SUFFIXES):
            return 'stopword'

        # 3. Чёрный список
        if any(pl in self.black_stopwords for pl in part_lemmas):
            return 'stop_black'

        # 4. Все нерим. части — стоп-слова
        if part_lemmas and all(
            is_roman_flags[i] or (pl in self.stopwords)
            for i, pl in enumerate(part_lemmas)
        ):
            return 'stopword'

        # 5. Префикс + основа в лексиконе
        if parts:
            p0 = LemmaNormalizer.normalize(parts[0]).strip('-')
            if p0 in self._strip_prefixes_set and len(parts) >= 2:
                base_token = ''.join(parts[1:])
                base_lemma = self._lemma_of(base_token)
                if (base_lemma in self.lexicon
                        or LemmaNormalizer.normalize(base_token) in self.lexicon):
                    return 'known'

        # 6. Все нерим. части в лексиконе
        if part_lemmas and all(
            is_roman_flags[i] or (pl in self.lexicon)
            for i, pl in enumerate(part_lemmas)
        ):
            return 'known'

        return 'unknown'

    # ------------------------------------------------------------------
    # Главный метод классификации
    # ------------------------------------------------------------------
    def classify(self, token: str) -> str:
        """
        Классифицирует токен относительно лексикона.

        Args:
            token: слово из текста (в оригинальном виде)

        Returns:
            'known'      — в лексиконе, не подсвечивать
            'unknown'    — не в лексиконе, подсветить красным
            'stop_black' — в чёрном списке, подсветить чёрным
            'stopword'   — служебное, не подсвечивать
        """
        if token in self._class_cache:
            return self._class_cache[token]

        token_nd    = self._strip_dashes(token)
        lemma       = self._lemma_of(token)
        tnorm       = LemmaNormalizer.normalize(token_nd)
        cand_lemmas = self._candidate_lemmas(token)

        def _cache_return(cat: str, reason: str) -> str:
            if self._dbg_on(token, lemma):
                logger.debug(
                    f"[classify] token='{token}' lemma='{lemma}' "
                    f"-> {cat} ({reason})"
                )
            self._class_cache[token] = cat
            return cat

        # --- 0. Чёрный список ---
        if lemma in self.black_stopwords or tnorm in self.black_stopwords:
            return _cache_return('stop_black', 'blacklist')

        # --- 1. Дефисные конструкции ---
        if self.HYPH_SPLIT_RE.search(token):
            parts = self._split_hyphen_parts(token)
            cat   = self.classify_hyphenated(parts)
            return _cache_return(cat, f'hyphen-parts={parts}')

        # --- 2. Римские цифры ---
        if self._is_roman(token_nd):
            return _cache_return('stopword', 'roman-numeral')

        # --- 3. Латиница ---
        if self._is_latin(token):
            if lemma in self.lexicon or tnorm in self.lexicon or (cand_lemmas & self.lexicon):
                return _cache_return('known', 'latin-in-lexicon')
            return _cache_return('unknown', 'latin-not-in-lexicon')

        # --- 4. Служебные / короткие / местоимения / NE ---
        is_service = (
            self._is_short(token_nd)
            or self._is_pronoun(token_nd)
            or self._is_function_word(token_nd)
            or self._is_force_green(token, lemma)
            or self._is_proper_strict(token)
            or (lemma in self.stopwords)
            or (tnorm in self.stopwords)
            or bool(cand_lemmas & self.stopwords)
        )
        if is_service:
            return _cache_return('stopword', 'service-or-stopword')

        # --- 5. В лексиконе ---
        if lemma in self.lexicon or tnorm in self.lexicon or bool(cand_lemmas & self.lexicon):
            return _cache_return('known', 'in-lexicon')

        # --- 6. Снятие приставок (слитное написание) ---
        for base in self._prefix_base_forms(token):
            base_lemma = self._lemma_of(base)
            if base_lemma in self.lexicon or LemmaNormalizer.normalize(base) in self.lexicon:
                return _cache_return('known', f'prefix-strip→{base}')

        # --- 7. Всё остальное ---
        return _cache_return('unknown', 'fallback')

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _dbg_on(self, token: str, lemma: Optional[str] = None) -> bool:
        if not token:
            return False
        t = LemmaNormalizer.normalize(self._strip_dashes(token))
        if self.debug_tokens and t in self.debug_tokens:
            return True
        if lemma and LemmaNormalizer.normalize(lemma) in self.debug_tokens:
            return True
        return False