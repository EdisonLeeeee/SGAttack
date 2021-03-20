import tensorflow as tf

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) > LooseVersion("1.14"):
    import tensorflow.compat.v1 as tf
if LooseVersion(tf.__version__) > LooseVersion("2.0"):
    tf.disable_v2_behavior()
import numpy as np
import scipy.sparse as sp
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import f1_score

import utils

spdot = tf.sparse.sparse_dense_matmul
dot = tf.matmul


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class GCN:
    def __init__(self, adj, x, y, hidden=16, name="",
                 with_relu=True, params_dict={'dropout': 0.5}, gpu_id=None,
                 seed=None):
        adj = utils.preprocess_adj(adj)
        num_features = x.shape[1]
        num_classes = y.max() + 1

        self.graph = tf.Graph()
        with self.graph.as_default():
            if seed:
                tf.set_random_seed(seed)

            with tf.variable_scope(name) as scope:
                w_init = glorot_uniform
                self.name = name

                self.dropout = params_dict. get('dropout', 0.)
                if not with_relu:
                    self.dropout = 0

                self.learning_rate = params_dict. get('learning_rate', 0.01)

                self.weight_decay = params_dict. get('weight_decay', 5e-4)
                self.N, self.D = x.shape

                self.node_ids = tf.placeholder(tf.int32, [None], 'node_ids')
                self.node_labels = tf.placeholder(tf.int32, [None, num_classes], 'node_labels')

                # bool placeholder to turn on dropout during training
                self.training = tf.placeholder_with_default(False, shape=())

                self.labels = np.eye(num_classes)[y]
                self.adj = tf.SparseTensor(*utils.sparse_to_tuple(adj))
                self.adj = tf.cast(self.adj, tf.float32)
                self.X_sparse = tf.SparseTensor(*utils.sparse_to_tuple(x))
                self.X_sparse = tf.cast(self.X_sparse, tf.float32)
                self.X_dropout = sparse_dropout(self.X_sparse, 1 - self.dropout,
                                                (int(self.X_sparse.values.get_shape()[0]),))
                # only use drop-out during training
                self.X_comp = tf.cond(self.training,
                                      lambda: self.X_dropout,
                                      lambda: self.X_sparse) if self.dropout > 0. else self.X_sparse

                self.W1 = tf.get_variable('W1', [self.D, hidden], tf.float32, initializer=w_init())
                self.b1 = tf.get_variable('b1', dtype=tf.float32, initializer=tf.zeros(hidden))

                self.h1 = spdot(self.adj, spdot(self.X_comp, self.W1))

                if with_relu:
                    self.h1 = tf.nn.relu(self.h1 + self.b1)

                self.h1_dropout = tf.nn.dropout(self.h1, rate=self.dropout)

                self.h1_comp = tf.cond(self.training,
                                       lambda: self.h1_dropout,
                                       lambda: self.h1) if self.dropout > 0. else self.h1

                self.W2 = tf.get_variable('W2', [hidden, num_classes], tf.float32, initializer=w_init())
                self.b2 = tf.get_variable('b2', dtype=tf.float32, initializer=tf.zeros(num_classes))

                self.logits = spdot(self.adj, dot(self.h1_comp, self.W2))
                if with_relu:
                    self.logits += self.b2
                self.logits_gather = tf.gather(self.logits, self.node_ids)

                self.predictions = tf.nn.softmax(self.logits_gather)

                self.loss_per_node = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_gather,
                                                                                labels=self.node_labels)
                self.loss = tf.reduce_mean(self.loss_per_node)

                # weight decay only on the first layer, to match the original implementation
                if with_relu:
                    self.loss += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in [self.W1, self.b1]])

                var_l = [self.W1, self.W2]
                if with_relu:
                    var_l.extend([self.b1, self.b2])
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                  var_list=var_l)

                self.varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                self.local_init_op = tf.variables_initializer(self.varlist)

                if gpu_id is None:
                    config = tf.ConfigProto(
                        device_count={'GPU': 0}
                    )
                else:
                    gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                    config = tf.ConfigProto(gpu_options=gpu_options)

                self.session = tf.Session(config=config)
                self.init_op = tf.global_variables_initializer()
                self.session.run(self.init_op)

    def get_weight(self):
        return self.session.run([self.W1, self.W2])

    def reset_weight(self):
        varlist = self.varlist
        self.session.run(self.local_init_op)

    def close(self):
        self.session.close()

    def convert_varname(self, vname, to_namespace=None):
        """
        Utility function that converts variable names to the input namespace.

        Parameters
        ----------
        vname: string
            The variable name.

        to_namespace: string
            The target namespace.

        Returns
        -------

        """
        namespace = vname.split("/")[0]
        if to_namespace is None:
            to_namespace = self.name
        return vname.replace(namespace, to_namespace)

    def set_variables(self, var_dict):
        """
        Set the model's variables to those provided in var_dict. This is e.g. used to restore the best seen parameters
        after training with patience.

        Parameters
        ----------
        var_dict: dict
            Dictionary of the form {var_name: var_value} to assign the variables in the model.

        Returns
        -------
        None.
        """

        with self.graph.as_default():
            if not hasattr(self, 'assign_placeholders'):
                self.assign_placeholders = {v.name: tf.placeholder(v.dtype, shape=v.get_shape()) for v in self.varlist}
                self.assign_ops = {v.name: tf.assign(v, self.assign_placeholders[v.name])
                                   for v in self.varlist}
            to_namespace = list(var_dict.keys())[0].split("/")[0]
            self.session.run(list(self.assign_ops.values()), feed_dict={val: var_dict[key]
                                                                        for key, val in self.assign_placeholders.items()})
#             self.session.run(list(self.assign_ops.values()), feed_dict={val: var_dict[self.convert_varname(key, to_namespace)]
#                                                                         for key, val in self.assign_placeholders.items()})

    def train(self, train_nodes, val_nodes, patience=30, n_iters=200, verbose=False, dump_best=True):
        early_stopping = patience

        best_performance = 0
        patience = early_stopping
        labels = self.labels
        feed = {self.node_ids: train_nodes,
                self.node_labels: labels[train_nodes]}
        if hasattr(self, 'training'):
            feed[self.training] = True
        for it in range(n_iters):
            _loss, _ = self.session.run([self.loss, self.train_op], feed)
            predict = self.predictions.eval(session=self.session, feed_dict={self.node_ids: val_nodes}).argmax(1)
            f1_micro, f1_macro = evaluate(predict, np.argmax(labels[val_nodes], 1))
            perf_sum = f1_micro + f1_macro
            if perf_sum > best_performance:
                best_performance = perf_sum
                patience = early_stopping
                if dump_best:
                    # var dump to memory is much faster than to disk using checkpoints
                    var_dump_best = {v.name: v.eval(self.session) for v in self.varlist}
            else:
                patience -= 1
            if it > early_stopping and patience <= 0:
                break
        if verbose:
            print('converged after {} iterations'.format(it - patience))
        if dump_best:
            # Put the best observed parameters back into the model
            self.set_variables(var_dump_best)

    def test(self, test_nodes):
        predict = self.predictions.eval(session=self.session, feed_dict={self.node_ids: test_nodes}).argmax(1)
        f1_micro, f1_macro = evaluate(predict, np.argmax(self.labels[test_nodes], 1))
        return f1_micro, f1_macro

    def predict(self, nodes):
        if np.isscalar(nodes):
            nodes = [nodes]
        pred = self.predictions.eval(session=self.session, feed_dict={self.node_ids: nodes})
        return pred.squeeze()


def evaluate(test_pred, test_real):
    return f1_score(test_real, test_pred, labels=np.unique(test_pred), average='micro'), f1_score(test_real, test_pred, labels=np.unique(test_pred), average='macro')
