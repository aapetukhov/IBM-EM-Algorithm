from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """    
    intersection = 0
    total_predicted = 0

    for ref, pred in zip(reference, predicted):
        possible_alignments = set(ref.sure) | set(ref.possible)
        predicted_alignments = set(pred)
        
        intersection += len(predicted_alignments & possible_alignments)
        total_predicted += len(pred)

    return intersection, total_predicted


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_sure: total number of sure alignments over all sentences
    """
    intersection = 0
    total_sure = 0

    for ref, pred in zip(reference, predicted):
        sure_alignments = set(ref.sure)
        predicted_alignments = set(pred)
        
        intersection += len(predicted_alignments & sure_alignments)
        total_sure += len(sure_alignments)

    return intersection, total_sure


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    precision_numerator, precision_denominator = compute_precision(reference, predicted)
    recall_numerator, recall_denominator = compute_recall(reference, predicted)

    return 1 - (precision_numerator + recall_numerator) / (precision_denominator + recall_denominator)
