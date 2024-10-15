import glob
import gzip
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from lxml import etree


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """

    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """

    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """

    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(
    filename: str,
) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """

    with open(filename, "r", encoding="utf-8") as file:
        content = file.read().replace("&", "&amp;")

    root = ET.fromstring(content)

    sentence_pairs = []
    alignments = []

    for sentences in root.findall("s"):
        eng_sentence = sentences.find("english").text.split()
        cz_sentence = sentences.find("czech").text.split()
        sentence_pairs.append(SentencePair(source=eng_sentence, target=cz_sentence))

        sure_pairs = sentences.find("sure").text
        sure_alignments = (
            [
                (int(pair.split("-")[0]), int(pair.split("-")[1]))
                for pair in sure_pairs.split()
            ]
            if sure_pairs
            else []
        )

        possible_text = sentences.find("possible").text
        possible_alignments = (
            [
                (int(pair.split("-")[0]), int(pair.split("-")[1]))
                for pair in possible_text.split()
            ]
            if possible_text
            else []
        )

        alignments.append(
            LabeledAlignment(sure=sure_alignments, possible=possible_alignments)
        )

    return sentence_pairs, alignments


def get_token_to_index(
    sentence_pairs: List[SentencePair], freq_cutoff=None
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    #"""

    source_counter = Counter()
    target_counter = Counter()

    for sentence_pair in sentence_pairs:
        source_counter.update(sentence_pair.source)
        target_counter.update(sentence_pair.target)

    source_dict = {}
    target_dict = {}

    for i, (token, _) in enumerate(source_counter.most_common(freq_cutoff)):
        source_dict[token] = i
    for i, (token, _) in enumerate(target_counter.most_common(freq_cutoff)):
        target_dict[token] = i

    return source_dict, target_dict


def tokenize_sents(
    sentence_pairs: List[SentencePair], source_dict, target_dict
) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.

    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []

    for sentence_pair in sentence_pairs:
        tokenized_source = [
            source_dict.get(token)
            for token in sentence_pair.source
            if token in source_dict
        ]
        tokenized_target = [
            target_dict.get(token)
            for token in sentence_pair.target
            if token in target_dict
        ]

        if not tokenized_source or not tokenized_target:
            continue

        tokenized_sentence_pairs.append(
            TokenizedSentencePair(
                source_tokens=np.array(tokenized_source, dtype=np.int32),
                target_tokens=np.array(tokenized_target, dtype=np.int32),
            )
        )

    return tokenized_sentence_pairs


############## ниже кастомные функции для бонусов


def unicode_normalize(
    sentence_pairs: List[SentencePair], encoding: str = "NFC"
) -> List[SentencePair]:
    """Applies Unicode Normalization to a list of SentencePair's.
    Also converts all strings to lowercase

    Args:
        sentence_pairs (List[SentencePair]): _description_
        encoding (str): normalization type (one of ['NFC', 'NFD', 'NFKD', 'NFKC'])

    Returns:
        List[SentencePair]: encoded sentence pairs
    """
    normalized_sentence_pairs = []

    for sentence_pair in sentence_pairs:
        normalized_sentence_pairs.append(
            SentencePair(
                source=[
                    unicodedata.normalize(encoding, word.lower())
                    for word in sentence_pair.source
                ],
                target=[
                    unicodedata.normalize(encoding, word.lower())
                    for word in sentence_pair.target
                ],
            )
        )
    return normalized_sentence_pairs


def parse_gzip(filename: str):
    """Парсинг XML файла с обработкой ошибок из gzip"""
    with gzip.open(filename, "rb") as f:
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(f, parser=parser)
    return tree


def extract_sentences_from_gzip(
    file_prefix: str, encoding: str = "NFC"
) -> List[SentencePair]:
    """Извлекаем список SentencePair-ов из тройки файлов, начинающихся с file_prefix"""
    salign_file = f"{file_prefix}-salign.xml.gz"
    en_file = f"{file_prefix}-en.xml.gz"
    cs_file = f"{file_prefix}-cs.xml.gz"

    # извлекаем выравнивания
    tree = parse_gzip(salign_file)
    alignments = []
    for pair in tree.iter("pair"):
        source_ids = [member.get("idref") for member in pair.find("members1")]
        target_ids = [member.get("idref") for member in pair.find("members2")]
        alignments.extend(zip(source_ids, target_ids))

    # извлекаем англ и чешские файлы из соответствющих файликов
    en_ids, cs_ids = zip(*alignments)
    en_sentences = {
        s.get("id"): " ".join(w.text for w in s.iter("w"))
        for s in parse_gzip(en_file).iter("s")
        if s.get("id") in en_ids
    }
    cs_sentences = {
        s.get("id"): " ".join(w.text for w in s.iter("w"))
        for s in parse_gzip(cs_file).iter("s")
        if s.get("id") in cs_ids
    }

    # сразу переводим в lowercase
    all_sentences = [
        SentencePair(
            source=unicodedata.normalize(encoding, en_sentences[en_id].lower()).split(),
            target=unicodedata.normalize(encoding, cs_sentences[cs_id].lower()).split(),
        )
        for en_id, cs_id in alignments
        if en_id in en_sentences and cs_id in cs_sentences
    ]

    return all_sentences


def extract_prefixes(directory: str) -> List[str]:
    """Извлекает уникальные префиксы файлов из заданной директории."""
    # обрезаем все файлнеймы до -14 символа (это длина -salign.xml.gz)
    return list(
        map(
            lambda filename: filename[:-14],
            glob.glob(f"{directory}/**/*-salign.xml.gz", recursive=True),
        )
    )


def parse_directory(directory: str, alpha=0.1) -> List[SentencePair]:
    prefixes = extract_prefixes(directory)
    prefixes = prefixes[: int(len(prefixes) * alpha)]
    sentences = []
    for prefix in prefixes:
        sentences.extend(extract_sentences_from_gzip(prefix))
    return sentences
