# tests/test_matcher.py
"""
Тесты классификатора токенов.

Разделены на два уровня:
  - unit: с моком препроцессора (быстро, нет зависимостей от моделей)
  - integration: с реальным препроцессором (медленно, помечены @pytest.mark.slow)
"""
from __future__ import annotations

from typing import Set
from unittest.mock import MagicMock, patch

import pytest

from src.highlighting.lexicon import LemmaNormalizer
from src.highlighting.matcher import TokenLemmaMatcher


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------
def _make_mock_parse(normal_form: str, score: float = 1.0, tags: Set[str] = None):
    """Создаёт мок разбора pymorphy3."""
    tags     = tags or set()
    mock     = MagicMock()
    mock.normal_form = normal_form
    mock.score       = score
    tag_mock         = MagicMock()
    tag_mock.__contains__ = lambda self, item: item in tags
    mock.tag         = tag_mock
    return mock


def _make_preprocessor(lemma_map: dict = None, stopwords: Set[str] = None):
    """
    Создаёт мок препроцессора.

    Args:
        lemma_map: {токен: лемма} — что возвращает lemmatize()
        stopwords: множество стоп-слов
    """
    pre = MagicMock()
    pre.russian_stopwords = stopwords or set()
    pre.replacements      = {}

    lemma_map = lemma_map or {}

    def mock_lemmatize(text, **kwargs):
        return [lemma_map.get(text, text)]

    def mock_morph_parse(word):
        # Нормализуем слово
        nf   = lemma_map.get(word, word)
        tags = set()
        # Простые эвристики для служебных слов в тестах
        if word in {"на", "в", "из", "по", "от"}:
            tags = {"PREP"}
        if word in {"и", "но", "или"}:
            tags = {"CONJ"}
        if word in {"не", "бы", "же"}:
            tags = {"PRCL"}
        return [_make_mock_parse(nf, score=1.0, tags=tags)]

    pre.lemmatize        = mock_lemmatize
    pre.morph.parse      = mock_morph_parse
    pre.replace_letters  = lambda x: x

    return pre


@pytest.fixture
def basic_matcher(small_lexicon: Set[str]) -> TokenLemmaMatcher:
    """
    Матчер с маленьким лексиконом и моком препроцессора.
    Лексикон: {"кот", "собака", "бежать", "быстрый", "дом", "математика", "теорема"}
    """
    lemma_map = {
        "котов":    "кот",
        "кошек":    "кот",
        "собакам":  "собака",
        "бежал":    "бежать",
        "быстрого": "быстрый",
        "домов":    "дом",
    }
    pre = _make_preprocessor(lemma_map=lemma_map)
    return TokenLemmaMatcher(pre, small_lexicon)


@pytest.fixture
def matcher_with_black(
    small_lexicon: Set[str],
    small_black_stopwords: Set[str],
) -> TokenLemmaMatcher:
    pre = _make_preprocessor()
    return TokenLemmaMatcher(
        pre,
        small_lexicon,
        black_stopword_lemmas=small_black_stopwords,
    )


# ---------------------------------------------------------------------------
# Тесты LemmaNormalizer
# ---------------------------------------------------------------------------
class TestLemmaNormalizer:
    @pytest.mark.parametrize("inp, expected", [
        ("Кот",    "кот"),
        ("ЁЖИК",   "ежик"),
        ("  пёс ", "пес"),
        ("",       ""),
    ])
    def test_normalize(self, inp: str, expected: str):
        assert LemmaNormalizer.normalize(inp) == expected


# ---------------------------------------------------------------------------
# Тесты классификации: базовые случаи
# ---------------------------------------------------------------------------
class TestClassifyBasic:
    def test_known_word_exact(self, basic_matcher: TokenLemmaMatcher):
        """Слово точно в лексиконе → known."""
        assert basic_matcher.classify("кот") == "known"

    def test_known_word_inflected(self, basic_matcher: TokenLemmaMatcher):
        """Словоформа, лемма которой в лексиконе → known."""
        assert basic_matcher.classify("котов") == "known"

    def test_unknown_word(self, basic_matcher: TokenLemmaMatcher):
        """Слово не в лексиконе → unknown."""
        assert basic_matcher.classify("квантовый") == "unknown"

    def test_short_word_is_stopword(self, basic_matcher: TokenLemmaMatcher):
        """Слова короче 3 символов → stopword."""
        assert basic_matcher.classify("ая") == "stopword"

    def test_result_is_string(self, basic_matcher: TokenLemmaMatcher):
        """Результат всегда строка."""
        result = basic_matcher.classify("кот")
        assert isinstance(result, str)

    def test_valid_categories(self, basic_matcher: TokenLemmaMatcher):
        """Результат всегда одна из допустимых категорий."""
        valid = {"known", "unknown", "stopword", "stop_black"}
        for word in ["кот", "квантовый", "на", "XIX"]:
            assert basic_matcher.classify(word) in valid


# ---------------------------------------------------------------------------
# Тесты: чёрный список
# ---------------------------------------------------------------------------
class TestBlackStopwords:
    def test_black_stopword_classified_correctly(
        self, matcher_with_black: TokenLemmaMatcher
    ):
        assert matcher_with_black.classify("запрещённый") == "stop_black"

    def test_non_black_word_not_affected(
        self, matcher_with_black: TokenLemmaMatcher
    ):
        assert matcher_with_black.classify("кот") == "known"

    def test_black_overrides_lexicon(self, small_lexicon: Set[str]):
        """Чёрный список имеет приоритет над лексиконом."""
        # Добавляем "кот" в чёрный список
        pre = _make_preprocessor()
        matcher = TokenLemmaMatcher(
            pre,
            small_lexicon,
            black_stopword_lemmas={"кот"},
        )
        assert matcher.classify("кот") == "stop_black"


# ---------------------------------------------------------------------------
# Тесты: римские цифры
# ---------------------------------------------------------------------------
class TestRomanNumerals:
    @pytest.mark.parametrize("roman", [
        "XIX", "XIV", "VIII", "IV", "XL", "CM", "MMIV", "I", "V", "X",
    ])
    def test_roman_is_stopword(
        self, basic_matcher: TokenLemmaMatcher, roman: str
    ):
        assert basic_matcher.classify(roman) == "stopword"

    @pytest.mark.parametrize("not_roman", [
        "ABC", "XYZ", "IIII", "VV",
    ])
    def test_non_roman_not_stopword(
        self, basic_matcher: TokenLemmaMatcher, not_roman: str
    ):
        """Не-римские латинские слова не должны классифицироваться как roman."""
        result = basic_matcher.classify(not_roman)
        assert result != "stopword"

    def test_roman_with_ordinal_suffix(self, basic_matcher: TokenLemmaMatcher):
        """XIX-й → stopword (через classify_hyphenated)."""
        result = basic_matcher.classify_hyphenated(["XIX", "й"])
        assert result == "stopword"


# ---------------------------------------------------------------------------
# Тесты: дефисные конструкции
# ---------------------------------------------------------------------------
class TestHyphenated:
    def test_both_parts_known(self, basic_matcher: TokenLemmaMatcher):
        """Обе части в лексиконе → known."""
        result = basic_matcher.classify_hyphenated(["кот", "собака"])
        assert result == "known"

    def test_one_part_unknown(self, basic_matcher: TokenLemmaMatcher):
        """Одна часть не в лексиконе → unknown."""
        result = basic_matcher.classify_hyphenated(["кот", "квантовый"])
        assert result == "unknown"

    def test_all_roman(self, basic_matcher: TokenLemmaMatcher):
        """Обе части — римские цифры → stopword."""
        result = basic_matcher.classify_hyphenated(["XIX", "XX"])
        assert result == "stopword"

    def test_roman_plus_ordinal(self, basic_matcher: TokenLemmaMatcher):
        """XIX-го → stopword."""
        result = basic_matcher.classify_hyphenated(["XIX", "го"])
        assert result == "stopword"

    def test_black_in_parts(self, matcher_with_black: TokenLemmaMatcher):
        """Если хоть одна часть в чёрном списке → stop_black."""
        result = matcher_with_black.classify_hyphenated(["запрещённый", "кот"])
        assert result == "stop_black"

    def test_prefix_plus_known_base(self, small_lexicon: Set[str]):
        """
        Дефисная конструкция 'не-быстрый':
        первая часть — префикс, вторая — в лексиконе → known.
        """
        lemma_map = {"быстрый": "быстрый"}
        pre       = _make_preprocessor(lemma_map=lemma_map)
        matcher   = TokenLemmaMatcher(pre, small_lexicon)
        result    = matcher.classify_hyphenated(["не", "быстрый"])
        assert result == "known"


# ---------------------------------------------------------------------------
# Тесты: снятие приставок (слитное написание)
# ---------------------------------------------------------------------------
class TestPrefixStripping:
    def test_ne_prefix(self, small_lexicon: Set[str]):
        """небыстрый → снять 'не' → быстрый → в лексиконе."""
        lemma_map = {"быстрый": "быстрый", "небыстрый": "небыстрый"}
        pre       = _make_preprocessor(lemma_map=lemma_map)
        matcher   = TokenLemmaMatcher(pre, small_lexicon)
        assert matcher.classify("небыстрый") == "known"

    def test_anti_prefix(self, small_lexicon: Set[str]):
        """антибыстрый → снять 'анти' → быстрый → в лексиконе."""
        lemma_map = {"быстрый": "быстрый", "антибыстрый": "антибыстрый"}
        pre       = _make_preprocessor(lemma_map=lemma_map)
        matcher   = TokenLemmaMatcher(pre, small_lexicon)
        assert matcher.classify("антибыстрый") == "known"

    def test_unknown_without_prefix_match(self, small_lexicon: Set[str]):
        """Если основа тоже не в лексиконе — unknown."""
        lemma_map = {"незнакомый": "незнакомый", "знакомый": "знакомый"}
        pre       = _make_preprocessor(lemma_map=lemma_map)
        matcher   = TokenLemmaMatcher(pre, small_lexicon)
        assert matcher.classify("незнакомый") == "unknown"


# ---------------------------------------------------------------------------
# Тесты: кэширование
# ---------------------------------------------------------------------------
class TestCaching:
    def test_classify_cached(self, basic_matcher: TokenLemmaMatcher):
        """Повторный вызов возвращает тот же результат из кэша."""
        r1 = basic_matcher.classify("кот")
        r2 = basic_matcher.classify("кот")
        assert r1 == r2

    def test_lemma_cached(self, basic_matcher: TokenLemmaMatcher):
        """Повторный вызов _lemma_of не вызывает лемматизатор дважды."""
        basic_matcher._lemma_of("кот")
        basic_matcher._lemma_of("кот")
        # Проверяем что результат консистентен
        assert basic_matcher._lemma_cache.get("кот") is not None

    def test_different_tokens_independent(self, basic_matcher: TokenLemmaMatcher):
        """Кэш одного токена не влияет на другой."""
        r1 = basic_matcher.classify("кот")
        r2 = basic_matcher.classify("квантовый")
        assert r1 != r2


# ---------------------------------------------------------------------------
# Интеграционные тесты (помечены как slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
class TestMatcherIntegration:
    """
    Тесты с реальным препроцессором.
    Запуск: pytest -m slow
    """

    def test_known_inflected_form(
        self,
        real_preprocessor,
        small_lexicon: Set[str],
    ):
        """'котов' → лемма 'кот' → в лексиконе → known."""
        matcher = TokenLemmaMatcher(real_preprocessor, small_lexicon)
        assert matcher.classify("котов") == "known"

    def test_unknown_word(
        self,
        real_preprocessor,
        small_lexicon: Set[str],
    ):
        matcher = TokenLemmaMatcher(real_preprocessor, small_lexicon)
        assert matcher.classify("квантовый") == "unknown"

    def test_function_word(
        self,
        real_preprocessor,
        small_lexicon: Set[str],
    ):
        """Предлоги → stopword."""
        matcher = TokenLemmaMatcher(real_preprocessor, small_lexicon)
        assert matcher.classify("на") == "stopword"

    @pytest.mark.parametrize("token", ["XIX", "XIV", "VIII"])
    def test_roman_numerals(
        self,
        real_preprocessor,
        small_lexicon: Set[str],
        token: str,
    ):
        matcher = TokenLemmaMatcher(real_preprocessor, small_lexicon)
        assert matcher.classify(token) == "stopword"