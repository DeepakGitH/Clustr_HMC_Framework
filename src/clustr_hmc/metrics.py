"""
Evaluation metrics for hierarchical classification.

"""
from contextlib import contextmanager

import numpy as np
from networkx import all_pairs_shortest_path_length, relabel_nodes
from sklearn.preprocessing import MultiLabelBinarizer

from clustr_hmc.constants import ROOT


@contextmanager
def multi_labeled(y_true, y_pred, graph):
    """
    Helper context manager for using the hierarchical evaluation metrics
    defined in this model.

    Briefly, the evaluation metrics expect data in a binarized multi-label format,
    the same as returned when using scikit-learn's MultiLabelBinarizer.

    This method therefore encapsulate the boilerplate required to fit such a
    label transformation on the data we wish to evaluate (y_true, y_pred) as well as
    applying it to the class hierarchy itself (graph), by relabeling the nodes.

    See the examples/classify_digits.py file for example usage.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, 1].
        ground truth targets

    y_pred : array-like, shape = [n_samples, 1].
        predicted targets

    graph : the class hierarchy graph, given as a `networkx.DiGraph` instance

    Returns
    -------
    y_true_ : array-like, shape = [n_samples, n_classes].
        ground truth targets, transformed to a binary multi-label matrix format.
    y_pred_ : array-like, shape = [n_samples, n_classes].
        predicted targets, transformed to a binary multi-label matrix format.
    graph_ : the class hierarchy graph, given as a `networkx.DiGraph` instance,
        transformed to use the (integer) IDs fitted by the multi label binarizer.

    """
    mlb = MultiLabelBinarizer()
    all_classes = [
        node
        for node in graph.nodes
        if node != ROOT
    ]
    # Nb. we pass a (singleton) list-within-a-list as fit() expects an iterable-of-iterables
    mlb.fit([all_classes])

    node_label_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(list(mlb.classes_))
    }

    yield (
        mlb.transform(y_true),
        mlb.transform(y_pred),
        relabel_nodes(graph, node_label_mapping),
    )


def fill_ancestors(y, graph, root, copy=True):
    """
    Compute the full ancestor set for y, where y is in binary multi-label format,
    e.g. as a matrix of 0-1.

    Each row will be processed and filled in with 1s in indexes corresponding
    to the ancestor nodes of those already marked with 1 in that row,
    based on the given class hierarchy graph.

    Parameters
    ----------
    y : array-like, shape = [n_samples, n_classes].
        multi-class targets, corresponding to graph node integer ids.

    graph : the class hierarchy graph, given as a `networkx.DiGraph` instance

    root : identifier of the (stub) root node of hierarchy

    copy : bool, whether to update the y array in-place. defaults to True.

    Returns
    -------
    y_ : array-like, shape = [n_samples, n_classes].
        multi-class targets, corresponding to graph node integer ids with
        all ancestors of existing labels in matrix filled in, per row.

    """
    y_ = y.copy() if copy else y
    paths = all_pairs_shortest_path_length(graph.reverse(copy=False))
    for target, distances in paths:
        if target == root:
            # Our stub root node, can skip
            continue
        ix_rows = np.where(y[:, target] > 0)[0]
        # all ancestors, except the last one which would be the root node
        ancestors = list(distances.keys())[:-1]
        y_[tuple(np.meshgrid(ix_rows, ancestors))] = 1
    graph.reverse(copy=False)
    return y_


def _flatten(x):
    t = []
    for y in x:
        t.extend(y)
    return t


def h_precision_score(y_true, y_pred):
    """
    Calculate the micro-averaged hierarchical precision ("hR") metric based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.

    Note that the format expected here for `y_true` and `y_pred` is a
    list of list y_true[0] = [['FMCG', 'SPICES'], ['FMCG', 'STAPLES']] and y_pred similar

    For motivation and definition details, see:

        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf

    Parameters
    ----------
    y_true : array-like, shape = [n_samples, n_classes].
        Ground truth multi-class targets.

    y_pred : array-like, shape = [n_samples, n_classes].
        Predicted multi-class targets.

    Returns
    -------
    hP : float
        The computed (micro-averaged) hierarchical precision score.

    """
    '''
    y_true_ = fill_ancestors(y_true, graph=class_hierarchy, root=root)
    y_pred_ = fill_ancestors(y_pred, graph=class_hierarchy, root=root)

    ix = np.where((y_true_ != 0) & (y_pred_ != 0))

    true_positives = len(ix[0])
    all_results = np.count_nonzero(y_pred_)

    '''
    assert len(y_true) == len(y_pred), "Length of actual and predicted should be same"
    true_set = [set(_flatten(x)) for x in y_true]
    pred_set = [set(_flatten(x)) for x in y_pred]
    sum_actual = sum([len(x) for x in true_set])
    sum_pred = sum([len(x) for x in pred_set])
    intersection = sum([len(x.intersection(y)) for x, y in zip(true_set, pred_set)])
    return float(intersection) / float(sum_pred)


def h_recall_score(y_true, y_pred):
    """
    Calculate the micro-averaged hierarchical recall ("hR") metric based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.

    Note that the format expected here for `y_true` and `y_pred` is a
    list of list y_true[0] = [['FMCG', 'SPICES'], ['FMCG', 'STAPLES']] and y_pred similar

    For motivation and definition details, see:

        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf

    Parameters
    ----------
    y_true : list of list, Assumes multi-label length of list n_samples
        Ground truth multi-label hierarchical targets.

    y_pred : list of list, Assumes multi-label length of list n_samples
        Predicted multi-label hierarchical targets.

    Returns
    -------
    hR : float
        The computed (micro-averaged) hierarchical recall score.

    """
    assert len(y_true) == len(y_pred), "Length of actual and predicted should be same"
    true_set = [set(_flatten(x)) for x in y_true]
    pred_set = [set(_flatten(x)) for x in y_pred]
    sum_actual = sum([len(x) for x in true_set])
    intersection = sum([len(x.intersection(y)) for x, y in zip(true_set, pred_set)])
    return float(intersection) / float(sum_actual)


def h_fbeta_score(y_true, y_pred, beta=1.):
    """
    Calculate the micro-averaged hierarchical F-beta ("hF_{\beta}") metric based on
    given set of true class labels and predicated class labels, and the
    class hierarchy graph.

    Note that the format expected here for `y_true` and `y_pred` is a
    list of list y_true = [['FMCG', 'SPICES'], ['FMCG', 'STAPLES']] and y_pred similar

    For motivation and definition details, see:

        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf

    Parameters
    ----------
    y_true : list of list, Assumes multi-label length of list n_samples
        Ground truth multi-label hierarchical targets.

    y_pred : list of list, Assumes multi-label length of list n_samples
        Predicted multi-label hierarchical targets.

    beta: float
        the beta parameter for the F-beta score. Defaults to F1 score (beta=1).

    Returns
    -------
    hFscore : float
        The computed (micro-averaged) hierarchical F-score.

    """
    hP = h_precision_score(y_true, y_pred)
    hR = h_recall_score(y_true, y_pred)
    return (1. + beta ** 2.) * hP * hR / (beta ** 2. * hP + hR)


def _filter_labels(y, node, level):
    y_filtered = []
    indices = []
    for i,y_row in enumerate(y):
        temp = []
        for y_label in y_row:
            if y_label[level] == node:
                temp.append(y_label[level:])
                indices.append(i)
        y_filtered.append(temp)
    return y_filtered, indices


def node_precision_recall_scores(y_true, y_pred, node, level):
    """
    Calculate the precision and recall of the given node

    Note that the format expected here for `y_true` and `y_pred` is a
    list of list y_true[0] = [['FMCG', 'SPICES'], ['FMCG', 'STAPLES']] and y_pred similar

    For motivation and definition details, see:

        Functional Annotation of Genes Using Hierarchical Text
        Categorization, Kiritchenko et al 2008

        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.68.5824&rep=rep1&type=pdf

    Parameters
    ----------
    y_true : list of list, Assumes multi-label length of list n_samples
        Ground truth multi-label hierarchical targets.

    y_pred : list of list, Assumes multi-label length of list n_samples
        Predicted multi-label hierarchical targets.

    node : Node on which statistics have to be calculated

    level: Level at which the node is (starts with 0 which is the base case in our case segment)

    Returns
    -------
    Precision, Recall : float
        The computed (micro-averaged) hierarchical precision score.

    """
    assert len(y_true) == len(y_pred), "Length of actual and predicted should be same"
    y_true_filter, true_ind = _filter_labels(y_true, node, level)
    y_pred_filter, pred_ind = _filter_labels(y_pred, node, level)
    intersection = len(set(true_ind).intersection(set(pred_ind)))
    return float(intersection) / float(len(pred_ind)), float(intersection) / float(len(true_ind))