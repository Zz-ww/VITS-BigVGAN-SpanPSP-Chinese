import dataclasses
from typing import List, Optional, Tuple

import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import tokenizations
import torch

from benepar import ptb_unescape
from benepar.parse_base import BaseInputExample
import transliterate


@dataclasses.dataclass
class ParsingExample(BaseInputExample):
    """A single parse tree and sentence."""

    words: List[str]
    space_after: List[bool]
    tree: Optional[nltk.Tree] = None
    _pos: Optional[List[Tuple[str, str]]] = None

    def leaves(self):
        if self.tree is not None:
            return self.tree.leaves()
        elif self._pos is not None:
            return [word for word, tag in self._pos]
        else:
            return None

    def pos(self):
        if self.tree is not None:
            return self.tree.pos()
        else:
            return self._pos

    def without_gold_annotations(self):
        return dataclasses.replace(self, tree=None, _pos=self.pos())


class Treebank(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    @property
    def trees(self):
        return [x.tree for x in self.examples]

    @property
    def sents(self):
        return [x.words for x in self.examples]

    @property
    def tagged_sents(self):
        return [x.pos() for x in self.examples]

    def filter_by_length(self, max_len):
        return Treebank([x for x in self.examples if len(x.leaves()) <= max_len])

    def without_gold_annotations(self):
        return Treebank([x.without_gold_annotations() for x in self.examples])


def read_text(text_path):
    sents = []
    sent = []
    end_of_multiword = 0
    multiword_combined = ""
    multiword_separate = []
    multiword_sp_after = False
    with open(text_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                if sent:
                    sents.append(([w for w, sp in sent], [sp for w, sp in sent]))
                    sent = []
                    assert end_of_multiword == 0
                continue
            fields = line.split("\t", 2)
            num_or_range = fields[0]
            w = fields[1]

            if "-" in num_or_range:
                end_of_multiword = int(num_or_range.split("-")[1])
                multiword_combined = w
                multiword_separate = []
                multiword_sp_after = "SpaceAfter=No" not in fields[-1]
                continue
            elif int(num_or_range) <= end_of_multiword:
                multiword_separate.append(w)
                if int(num_or_range) == end_of_multiword:
                    _, separate_to_combined = tokenizations.get_alignments(
                        multiword_combined, multiword_separate
                    )
                    have_up_to = 0
                    for i, char_idxs in enumerate(separate_to_combined):
                        if i == len(multiword_separate) - 1:
                            word = multiword_combined[have_up_to:]
                            sent.append((word, multiword_sp_after))
                        elif char_idxs:
                            word = multiword_combined[have_up_to : max(char_idxs) + 1]
                            sent.append((word, False))
                            have_up_to = max(char_idxs) + 1
                        else:
                            sent.append(("", False))
                    assert int(num_or_range) == len(sent)
                    end_of_multiword = 0
                    multiword_combined = ""
                    multiword_separate = []
                    multiword_sp_after = False
                continue
            else:
                assert int(num_or_range) == len(sent) + 1
                sp = "SpaceAfter=No" not in fields[-1]
                sent.append((w, sp))
    return sents


def load_trees(const_path, text_path=None, text_processing="default"):
    reader = BracketParseCorpusReader("", [const_path])
    trees = reader.parsed_sents()

    if text_path is not None:
        sents = read_text(text_path)
    elif text_processing in ("arabic-translit", "hebrew-translit"):
        translit = transliterate.TRANSLITERATIONS[
            text_processing.replace("-translit", "")
        ]
        sents = []
        for tree in trees:
            words = [translit(word) for word in tree.leaves()]
            sp_after = [True for _ in words]
            sents.append((words, sp_after))
    elif text_processing in ("arabic", "hebrew"):
        sents = []
        for tree in trees:
            words = tree.leaves()
            sp_after = [True for _ in words]
            sents.append((words, sp_after))
    elif text_processing == "chinese":
        sents = []
        for tree in trees:
            words = tree.leaves()
            sp_after = [False for _ in words]
            sents.append((words, sp_after))
    elif text_processing == "default":
        sents = []
        for tree in trees:
            words = ptb_unescape.ptb_unescape(tree.leaves())
            sp_after = ptb_unescape.guess_space_after(tree.leaves())
            sents.append((words, sp_after))
    else:
        raise ValueError(f"Bad value for text_processing: {text_processing}")

    assert len(trees) == len(sents)
    treebank = Treebank(
        [
            ParsingExample(tree=tree, words=words, space_after=space_after)
            for tree, (words, space_after) in zip(trees, sents)
        ]
    )
    for example in treebank:
        assert len(example.words) == len(example.leaves()), (
            "Constituency tree has a different number of tokens than the CONLL-U or "
            "other file used to specify reversible tokenization."
        )
    return treebank
