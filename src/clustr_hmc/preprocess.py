
import networkx as nx
from clustr_hmc import graph
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse.csr import csr_matrix


class DataPreprocess():

    def __init__(self, type):
        self.type = type

    def preprocess_label(self, root, G, labels):
        '''
        Here labels are provided only for the leaf node. The function
        :param root: Root Node
        :param G: the taxonomy hierarchy
        :param labels: The labels, here only the leaf node is available, assumes a list
        :return: labels with entire path
        '''
        if G:
            path_labels = []
            for label in labels:
                paths = []
                for node in label:
                    if G.has_node(node):
                        paths.extend(list(nx.all_simple_paths(G, source=root, target=node)))
                    else:
                        raise ValueError('Labels have issues. Label not found in Graph')
                path_labels.append(paths)
            return path_labels
        else:
            raise ValueError('Graph Not Found')

    def assign_nodes(self, G, labels):
        '''
        This returns the indices of rows which belong to each node. The Graph input is updated
        for each Node indices 'X' are added
        :param G: Input Taxonomy hierarchy
        :param labels: Here the labels are full paths (multiple paths in case of DAG). A list of lists
                        [[0,1,2],[0,4,5]]
        :return: The Graph is updated
        '''
        if G and labels:
            for i,label in enumerate(labels):
                for path in label:
                    for node in path:
                        if G.has_node(node):
                            if 'X' in G.nodes[node]:
                                if i not in G.nodes[node]['X']:
                                    G.nodes[node]['X'].append(i)
                            else:
                                G.nodes[node]['X'] = [i]
                        else:
                            raise ValueError('Node not in Taxonomy - '+str(node))
            return G
        else:
            raise ValueError('Graph or Labels empty')

    def _get_most_specific_for_node(self, G, node):
        '''
        This finds data point indices which have node as the most specific label
        :param G: The taxonomy graph is developed with 'X' for each node
        :param node: the node for which we need sampling
        :return: indices of data points
        '''
        if G:
            if G.has_node(node) and ('X' in G.nodes[node]):
                specific = []
                for p in G.successors(node):
                    specific.extend(G.nodes[p]['X'])
                return list(set(G.nodes[node]['X']) - set(specific))
            else:
                raise ValueError('Node not in graph or X not updated')

    def _get_all_labels_in_pred(self, G, node):
        '''
        This finds data point indices which have node as the most specific label
        :param G: The taxonomy graph is developed with 'X' for each node
        :param node: the node for which we need sampling
        :return: indices of data points
        '''
        if G:
            if G.has_node(node) and ('X' in G.nodes[node]):
                root = []
                for r in graph.root_nodes(G):
                    root.append(r)
                root = root[0]
                paths = list(nx.all_simple_paths(G, source=root, target=node))
                top_parents_ind = []
                for p in paths:
                    if len(p) < 2:
                        continue
                    top_parents_ind.extend(G.nodes[p[1]]['X'])
                return list(set(top_parents_ind))
            else:
                raise ValueError('Node not in graph or X not updated')
        else:
            raise ValueError('Unknown G')

    def _get_all_training(self, G):
        '''
        Generate indices of all training examples
        :param G: Taxonomy with X updated
        :return: list of indices of all training examples
        '''
        root = []
        for r in graph.root_nodes(G):
            root.append(r)
        if len(root) == 1:
            return G.nodes[root[0]]['X']
        else:
            raise ValueError('Not a single root')

    def _exclusive_policy(self, G, node):
        '''
        creates data set with exclusive policy for LCN classifier essentially
        add reference
        :param G: Taxonomy with X updated
        :param node: Node on which classifier needs to be built
        :return: X+, X_
        '''
        try:
            X_pos = self._get_most_specific_for_node(G, node)
            X_neg = list(set(self._get_all_training(G)) - set(X_pos))
            return X_pos, X_neg
        except ValueError as type:
            raise ValueError

    def _less_exclusive_policy(self, G, node):
        '''
        creates data set with less exclusive policy for LCN classifier essentially
        add reference
        :param G: Taxonomy with X updated
        :param node: Node on which classifier needs to be built
        :return: X+, X_
        '''
        try:
            X_pos = self._get_most_specific_for_node(G, node)
            X_neg = list(set(self._get_all_training(G)) - set(G.nodes[node]['X']))
            return X_pos, X_neg
        except ValueError as type:
            raise ValueError

    def _less_inclusive_policy(self, G, node):
        '''
        creates data set with less inclusive policy for LCN classifier essentially
        add reference
        :param G: Taxonomy with X updated
        :param node: Node on which classifier needs to be built
        :return: X+, X_
        '''
        try:
            X_pos = G.nodes[node]['X']
            X_neg = list(set(self._get_all_training(G)) - set(X_pos))
            return X_pos, X_neg
        except ValueError as type:
            raise ValueError

    def _inclusive_policy(self, G, node):
        '''
        creates data set with inclusive policy for LCN classifier essentially
        add reference
        :param G: Taxonomy with X updated
        :param node: Node on which classifier needs to be built
        :return: X+, X_
        '''
        try:
            X_pos = G.nodes[node]['X']
            X_neg = list(set(self._get_all_training(G)) - set(self._get_all_labels_in_pred(G, node)))
            return X_pos, X_neg
        except ValueError as type:
            raise ValueError

    def _sibling_policy(self, G, node):
        '''
        creates data set with sibling policy for LCN classifier essentially
        add reference
        :param G: Taxonomy with X updated
        :param node: Node on which classifier needs to be built
        :return: X+, X_
        '''
        try:
            X_pos = G.nodes[node]['X']
            siblings = graph.find_siblings_dag(G, node)
            X_neg = []
            for s in siblings:
                X_neg.extend(G.nodes[s]['X'])
            X_neg = list(set(X_neg))
            return X_pos, X_neg
        except ValueError as type:
            raise ValueError

    def _exclusive_siblings_policy(self, G, node):
        '''
        creates data set with exclusive policy for LCN classifier essentially
        add reference
        :param G: Taxonomy with X updated
        :param node: Node on which classifier needs to be built
        :return: X+, X_
        '''
        try:
            X_pos = self._get_most_specific_for_node(G, node)
            siblings = graph.find_siblings_dag(G, node)
            X_neg = []
            for s in siblings:
                X_neg.extend(self._get_most_specific_for_node(G, s))
            X_neg = list(set(X_neg))
            return X_pos, X_neg
        except ValueError as type:
            raise ValueError

    #def _sampling_policy(self, sample_ind, ):

    def generate_samples_for_local_classification(self, G, X, nodes, **kwargs):
        '''
        This function should be generic enough to create data for every run of the local classifier
        :param G: The taxonomy graph. This is a graph which has been updated where the nodes have 'X'
        :param X: Input Data features -> 2-D vector or matrix
        :param nodes: Nodes on which classifier will run. If these are multiple Nodes they are assumed to be multi-class
        multi-label. If this is only a single node than it is assumed that
        it is a binary classification. A scheme has to be provided to sample negative data else data is picked
        from siblings.
        WHAT HAPPENS TO EXAMPLES WHICH LETS SAY STOP AT A NODE AND DONT GO FURTHER.
        :param kwargs: Various parameters including sample generation scheme. Sampling if sampling needs to be done
        :return: X_ and Y_ where X_ is a subset of X and Y_ are labels. For Binary Classification Y_ will be Node and
        NOT_NODE
        '''
        if G and nodes and (X is not None):
            if isinstance(nodes, list):
                if len(nodes) == 1 and self.type == 'lcn':
                    if G.has_node(nodes[0]):
                        if 'X' in G.nodes[nodes[0]]:
                            try:
                                if 'policy' in kwargs:
                                    X_pos, X_neg = kwargs['policy'](G, nodes[0])
                                else:
                                    X_pos, X_neg = self._sibling_policy(G, nodes[0])
                                ## Will add sampling part later
                                X_neg = list(set(X_neg) - set(X_pos))
                                X_ = np.vstack([X[X_pos], X[X_neg]])
                                Y_ = [nodes[0]]*len(X_pos) + ['NOT_'+nodes[0]]*len(X_neg)
                                return X_, Y_
                            except ValueError as type:
                                raise ValueError
                        else:
                            raise ValueError('G Not updated yet with X values')
                    else:
                        raise ValueError('Node not in G')
                else:
                    if self.type == 'lcn':
                        raise ValueError('Please only provide single node for LCN type learning')
                    else:
                        if ('include_parent' in kwargs) and kwargs['include_parent']:
                            parents = []
                            for node in nodes:
                                if G.has_node(node):
                                    parents.extend([p for p in G.predecessors(node)])
                                else:
                                    raise ValueError('Node not in G')
                            parents = list(set(parents))
                            parent_X = []
                            for p in parents:
                                parent_X.extend(self._get_most_specific_for_node(G, p))
                            label_dict_multi = dict()
                            for node in nodes:
                                x_ind = G.nodes[node]['X']
                                for i in x_ind:
                                    if i in label_dict_multi:
                                        label_dict_multi[i].append(node)
                                    else:
                                        label_dict_multi[i] = [node]
                            Y_ = []
                            X_ind = []
                            for k,v in label_dict_multi.items():
                                Y_.append(v)
                                X_ind.append(k)
                            X_ = X[X_ind]
                            if len(parent_X) == 0:
                                X_ = np.vstack([X_, X[parent_X]])
                                Y_ = Y_ + ['NONE']*len(parent_X)
                        else:
                            label_dict_multi = dict()
                            for node in nodes:
                                x_ind = G.nodes[node]['X']
                                for i in x_ind:
                                    if i in label_dict_multi:
                                        label_dict_multi[i].append(node)
                                    else:
                                        label_dict_multi[i] = [node]
                            Y_ = []
                            X_ind = []
                            for k, v in label_dict_multi.items():
                                Y_.append(v)
                                X_ind.append(k)
                            X_ = X[X_ind]
                        return X_, Y_
            elif isinstance(nodes, str):
                if G.has_node(nodes):
                    if 'X' in G.nodes[nodes]:
                        try:
                            if 'policy' in kwargs:
                                X_pos, X_neg = kwargs['policy'](G, nodes)
                            else:
                                X_pos, X_neg = self._sibling_policy(G, nodes)
                            ## Will add sampling part later
                            X_neg = list(set(X_neg) - set(X_pos))
                            X_ = np.vstack([X[X_pos], X[X_neg]])
                            Y_ = [nodes] * len(X_pos) + ['NOT_' + nodes] * len(X_neg)
                            return X_, Y_
                        except ValueError as type:
                            raise ValueError
                    else:
                        raise ValueError('G Not updated yet with X values')
                else:
                    raise ValueError('Node not in G')
            else:
                raise ValueError('nodes variable not understood')
        else:
            raise ValueError('Values missing')

    def select_features(self, X, y):
        """
        Perform feature selection for training data.
        Can be overridden by a sub-class to implement feature selection logic.
        """
        return X


class LabelTransformer():

    def __init__(self, multilabel=True, binary=False, **kwargs):
        self.multilabel = multilabel
        if 'transformer' in kwargs:
            self.transformer = kwargs['transformer']
        else:
            if multilabel:
                self.transformer = MultiLabelBinarizer()
            else:
                self.transformer = OneHotEncoder()

    def fit(self, labels):
        self.transformer.fit(labels)

    def transform(self, labels):
        temp = self.transformer.transform(labels)
        if isinstance(temp, csr_matrix):
            return temp.toarray()
        else:
            return temp



