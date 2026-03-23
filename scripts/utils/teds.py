"""TEDS (Tree Edit Distance based Similarity) metric for table evaluation.

Adapted from OmniDocBench (https://github.com/opendatalab/OmniDocBench).
Original: Copyright 2020 IBM, Apache 2.0 License.

TEDS compares predicted and ground-truth HTML tables as trees using APTED.
Two modes: full TEDS (structure + content) and TEDS-struct (structure only).
"""

import logging
from typing import List

import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from collections import deque

logger = logging.getLogger(__name__)


class TableTree(Tree):
    """Tree node representing an HTML table element for APTED."""

    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag, self.colspan, self.rowspan, self.content
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    """APTED cost configuration for table tree edit operations."""

    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Computes Tree Edit Distance based Similarity between HTML tables."""

    def __init__(self, structure_only: bool = False, ignore_nodes: List[str] = None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenize HTML node content into characters and tags."""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Convert lxml HTML tree to TableTree for APTED."""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())

        if parent is not None:
            parent.children.append(new_node)

        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)

        if parent is None:
            return new_node

    def evaluate(self, pred: str, true: str) -> float:
        """Compute TEDS score between predicted and ground-truth HTML tables.

        Returns float in [0, 1]: 1.0 = identical, 0.0 = completely different.
        """
        if (not pred) or (not true):
            return 0.0

        try:
            parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
            pred_tree = html.fromstring(pred, parser=parser)
            true_tree = html.fromstring(true, parser=parser)
        except Exception as e:
            logger.warning(f"HTML parsing failed: {e}")
            return 0.0

        pred_tables = pred_tree.xpath("body/table")
        true_tables = true_tree.xpath("body/table")

        if not pred_tables or not true_tables:
            return 0.0

        pred_table = pred_tables[0]
        true_table = true_tables[0]

        if self.ignore_nodes:
            etree.strip_tags(pred_table, *self.ignore_nodes)
            etree.strip_tags(true_table, *self.ignore_nodes)

        n_nodes_pred = len(pred_table.xpath(".//*"))
        n_nodes_true = len(true_table.xpath(".//*"))
        n_nodes = max(n_nodes_pred, n_nodes_true)

        if n_nodes == 0:
            return 0.0

        tree_pred = self.load_html_tree(pred_table)
        tree_true = self.load_html_tree(true_table)
        distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()

        return 1.0 - (float(distance) / n_nodes)


# Cached TEDS instances
_teds_full = TEDS(structure_only=False, ignore_nodes=["thead", "tbody"])
_teds_struct = TEDS(structure_only=True, ignore_nodes=["thead", "tbody"])


def compute_teds(pred_html: str, gt_html: str) -> float:
    """Compute full TEDS (structure + content)."""
    return _teds_full.evaluate(pred_html, gt_html)


def compute_teds_struct(pred_html: str, gt_html: str) -> float:
    """Compute TEDS-struct (structure only, ignoring cell content)."""
    return _teds_struct.evaluate(pred_html, gt_html)


def compute_teds_batch(
    predictions: List[str], ground_truths: List[str], structure_only: bool = False
) -> List[float]:
    """Batch TEDS computation for list of (prediction, ground_truth) pairs."""
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Lists must have equal length: {len(predictions)} vs {len(ground_truths)}"
        )

    evaluator = _teds_struct if structure_only else _teds_full
    scores = []
    for pred, gt in zip(predictions, ground_truths):
        try:
            score = evaluator.evaluate(pred, gt)
        except Exception as e:
            logger.warning(f"TEDS evaluation failed: {e}. Assigning score 0.0")
            score = 0.0
        scores.append(score)

    return scores
