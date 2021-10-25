import re

from typing import Iterable

from ..utils import run


@run
def drop_not_alpha(tokens: Iterable) -> list:
    """
    Function for dropping non alphabetic tokens
    """
    return [token for token in tokens if token.is_alpha]


@run
def drop_stop_words(tokens: Iterable) -> list:
    """
    Function for dropping stop words
    """
    return [token for token in tokens if not token.is_stop]


@run
def drop_no_vector(tokens: Iterable) -> list:
    """
    Function for dropping tokens without vectors
    """
    return [token for token in tokens if not token.is_oov]


@run
def drop_character(tokens: Iterable) -> list:
    """
    Function for dropping single characters
    """
    return [token for token in tokens if len(token.text) > 1]


@run
def drop_fully_consonants(tokens: Iterable) -> list:
    """
    Function for dropping tokens that fully consist of consonant characters
    """
    reg = re.compile(r'^[bcdfghjklmnpqrstvwxyz]+$')
    return [token for token in tokens if not reg.match(token.text)]


@run
def drop_fully_vowels(tokens: Iterable) -> list:
    """
    Function for dropping tokens that fully consist of vowel characters
    """
    reg = re.compile(r'^[aeiou]+$')
    return [token for token in tokens if not reg.match(token.text)]
