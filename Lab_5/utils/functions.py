#!/bin/python
import xml.etree.ElementTree as ET
import os
import sys
import nltk
from nltk.tokenize import WhitespaceTokenizer
from .models import Token, Entity

# Store as a list of dictionaries the word, the offset interval and the label (drug, group, brand, drug_n,...) of each entity in the sentence.
def get_entities(sentence):
    entities = []
    for ent in sentence.findall('entity'):
        entity = Entity(**ent.attrib)
        entities.append(entity)
    return entities

def tokenize(sentence):
    span_generator = WhitespaceTokenizer().span_tokenize(sentence)
    tokens = [(sentence[span[0]: span[1]], span[0], span[1] - 1) for span in span_generator]

    new_tokens = []
    for i, token in enumerate(tokens):
        word, offset_from, offset_to = token

        if (len(word) > 1) and (word.endswith(',') or word.endswith('.') or word.endswith(':') or word.endswith(';')):
            punct = word[-1]
            punct_offset_from = offset_to
            punct_offset_to = offset_to

            word = word[:-1]
            offset_to -= 1

            new_tokens.append(Token(word, offset_from, offset_to))
            new_tokens.append(Token(punct, punct_offset_from, punct_offset_to))

        elif (len(word) > 1) and word[0] == '(' and (word[0:2] != '(+' or word[0:2] != '(-'):
            punct = word[0]
            punct_offset_from = offset_from
            punct_offset_to = offset_from

            word = word[1:]
            offset_from += 1

            new_tokens.append(Token(punct, punct_offset_from, punct_offset_to))
            new_tokens.append(Token(word, offset_from, offset_to))
        else:
            new_tokens.append(Token(word, offset_from, offset_to))

    return new_tokens

def detect_label(token, entities):
    for entity in entities:
        # If the two offsets are equal, then it corresponds to the same word and type.
        if token.offset_from == entity.offset_from and token.offset_to == entity.offset_to:
            token.type = "B-" + entity.type
            return
        # If the token offset interval is inside the entity offset interval, then it is a first or the continuation of a type sequence.
        elif entity.offset_from <= token.offset_from and token.offset_to <= entity.offset_to:
            if entity.offset_from == token.offset_from:
                token.type = "B-" + entity.type
            else:
                token.type = "I-" + entity.type

            return
        else:
            token.type = "O"

