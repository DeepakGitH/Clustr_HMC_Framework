
from clustr_hmc.constants import (
    CLASSIFIER,
    DEFAULT,
    METAFEATURES,
    ROOT,
    LABEL_TRANSFORMER,
)
from clustr_hmc import preprocess as pre
from clustr_hmc.validation import is_estimator, validate_parameters
from clustr_hmc.dummy import DummyProgress
from sklearn.dummy import DummyClassifier
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    clone,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


class TrainTraverse():

    def __init__(self, algo, G, X, local_model, path_labels, is_tree, is_multi_label, do_transform=True, label_transformer=None, include_parent=False, policy=None, mlb_model=None, early_stopping=False, threshold=10):
        self.graph_ = G
        self.X = X
        self.labels = path_labels
        self.model = local_model
        self.prep = pre.DataPreprocess(algo)
        self.algorithm = algo
        self.is_tree = is_tree
        self.is_multi_label = is_multi_label
        self.label_transformer = label_transformer
        self.do_transform = do_transform
        self.include_parent = include_parent
        self.policy = policy
        self.mlb_model = mlb_model
        self.early_stopping = early_stopping
        self.threshold = threshold

    def training_propagation(self, type='lcpn', **kwargs):
        '''
        Traverses the graph for LCPN propagation and creates the training for each local model and trains the local
        model
        :param G: Taxonomay Graph with node indices added
        :param X: Feature matrix
        :param path_labels: Labels with entire paths available
        :param local_model: The local classification model
        :return: Updated Graph with Trained Model
        '''
        if type in ['lcpn', 'lcn']:
            progress = DummyProgress()
            self._recursive_train_local_classifiers(ROOT, progress)

    def _recursive_train_local_classifiers(self, node_id, progress):
        if CLASSIFIER in self.graph_.nodes[node_id]:
            # Already trained classifier at this node, skip
            return
        if self.early_stopping:
            if not self._filter_by_size(node_id, self.threshold):
                exit()
        progress.update(1)
        self._train_local_classifier(node_id)
        for child_node_id in self.graph_.successors(node_id):
            self._recursive_train_local_classifiers(node_id=child_node_id,progress=progress)

    def _train_local_classifier(self, node_id):
        if self.graph_.out_degree(node_id) == 0:
            # Leaf node
            if self.algorithm == "lcpn":
                # Leaf nodes do not get a classifier assigned in LCPN algorithm mode.
                return
        if self.algorithm == "lcpn":
            nodes = [node for node in self.graph_.successors(node_id)]
            X_train, Y_train = self.prep.generate_samples_for_local_classification(G=self.graph_, X=self.X,
                                                                                   nodes=nodes, include_parent=self.include_parent)
            if self.do_transform:
                if self.is_tree:
                    if self.label_transformer is None:
                        lt = pre.LabelTransformer(multilabel=False)
                        Y_ = lt.fit(Y_train).transform(Y_train)
                    else:
                        lt = pre.LabelTransformer(multilabel=False, transformer=self.label_transformer)
                        Y_ = lt.fit(Y_train).transform(Y_train)
                else:
                    if self.label_transformer is None:
                        lt = pre.LabelTransformer(multilabel=True)
                        Y_ = lt.fit(Y_train).transform(Y_train)
                    else:
                        lt = pre.LabelTransformer(multilabel=True, transformer=self.label_transformer)
                        Y_ = lt.fit(Y_train).transform(Y_train)
                self.graph_.nodes[node_id][LABEL_TRANSFORMER] = lt
            else:
                Y_ = Y_train
        elif self.algorithm == "lcn":
            if self.policy is None:
                X_train, Y_train = self.prep.generate_samples_for_local_classification(G=self.graph_, X=self.X,
                                                                                        nodes=node_id)
            else:
                X_train, Y_train = self.prep.generate_samples_for_local_classification(G=self.graph_, X=self.X,
                                                                                        nodes=node_id, policy=self.policy)
            if self.do_transform:
                if self.label_transformer is None:
                    lt = pre.LabelTransformer(multilabel=False)
                    Y_ = lt.fit(Y_train).transform(Y_train)
                else:
                    lt = pre.LabelTransformer(multilabel=False, transformer=self.label_transformer)
                    Y_ = lt.fit(Y_train).transform(Y_train)
                self.graph_.nodes[node_id][LABEL_TRANSFORMER] = lt
            else:
                Y_ = Y_train
        targets = set([y_l for y_d in Y_train for y_l in y_d])
        X_ = self.prep.select_features(X_train,Y_train)
        if len(targets) == 1:
            # Training data could be materialized for only a single target at current node
            constant = list(targets)[0]
            clf = DummyClassifier(strategy="constant", constant=constant)
        else:
            if self.algorithm in ['lcpn', 'lcpl']:
                # TODO: Bring this out probably as a Multi Label Strategy Parameter
                if self.mlb_model is None and (self.is_tree or self.is_multi_label):
                    clf = OneVsRestClassifier(self._base_estimator_for(node_id))
                elif self.mlb_model is not None:
                    clf = self.mlb_model(self._base_estimator_for(node_id))
            else:
                clf = self._base_estimator_for(node_id)
        clf.fit(X=X_, y=Y_)
        self.graph_.nodes[node_id][CLASSIFIER] = clf

    def _base_estimator_for(self, node_id):
        base_estimator = None
        if self.model is None:
            # No base estimator specified by user, try to pick best one
            base_estimator = self._make_base_estimator(node_id)
        elif isinstance(self.model, dict):
            # User provided dictionary mapping nodes to estimators
            if node_id in self.model:
                base_estimator = self.model[node_id]
            else:
                base_estimator = self.model[DEFAULT]
        elif is_estimator(self.model):
            # Single base estimator object, return a copy
            base_estimator = self.model
        else:
            # By default, treat as callable factory
            base_estimator = self.model(node_id=node_id, graph=self.graph_)
        return clone(base_estimator)

    def _make_base_estimator(self, node_id):
        """Create a default base estimator if a more specific one was not chosen by user."""
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
        )

    def _filter_by_size(self, node_id, threshold):
        if self.graph_.nodes[node_id][METAFEATURES]['n_samples'] < threshold:
            return False
        else:
            return True

