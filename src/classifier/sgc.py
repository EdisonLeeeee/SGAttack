import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) > LooseVersion("1.14"):
    import tensorflow.compat.v1 as tf
    
from time import perf_counter
from sklearn.metrics import f1_score

from utils import preprocess_adj

def sgc_precompute(features, adj, K=2):
    adj = preprocess_adj(adj).tocoo()
    features = features.tocoo()
    t = perf_counter()
    for _ in range(K):
        features = adj.dot(features)
    precompute_time = perf_counter() - t

    return features, precompute_time

class SGC:
    def __init__(self, adj, x, y, K=2, seed=42,
                use_bias=True, learning_rate=0.1,
                weight_decay=2e-5, dropout=0.):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if seed:
                tf.set_random_seed(seed)

            num_features=x.shape[1]
            num_classes = y.max() + 1
            self.x = tf.placeholder(tf.float32, shape=(None, num_features))
            self.y = tf.placeholder(tf.float32, shape=(None, num_classes))
            self.dropout = tf.placeholder_with_default(dropout, shape=())
            self.input_dim = num_features
            self.output_dim = num_classes

            self.outputs = tf.layers.Dense(units=self.output_dim, use_bias=use_bias,  
                                           kernel_initializer=tf.glorot_uniform_initializer, 
                                           bias_initializer=tf.zeros_initializer, 
                                           kernel_regularizer=tf.nn.l2_loss)(self.x)
            
            self.outputs = tf.nn.dropout(self.outputs, rate=self.dropout)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.y)) + weight_decay*tf.losses.get_regularization_loss()
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

            correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.variables = tf.trainable_variables()
            self.sess = tf.Session(graph=self.graph)
            self.run(tf.global_variables_initializer())

            self.labels = np.eye(num_classes)[y]
            features, _ = sgc_precompute(x, adj)
            self.features = features.A

    def train(self, train_nodes, val_nodes=None, epochs=200,
                verbose=True, patience=50, early_stop=True):
        t = perf_counter()

        early_stopping = patience
        best_performance = 0

        x_train = self.features[train_nodes]
        y_train = self.labels[train_nodes]
        if val_nodes is not None:
            x_val= self.features[val_nodes]
            y_val = self.labels[val_nodes]


        for epoch in range(epochs):
            # Training step
            feed_dict = self.construct_feed_dict(x=x_train, y=y_train)
            _, avg_loss, avg_accuracy = self.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)

            if early_stop and val_nodes is not None:
                f1_micro, f1_macro = self.test_f1(x_val, y_val)
                perf_sum = f1_micro + f1_macro
                if perf_sum > best_performance:
                    best_performance = perf_sum
                    patience = early_stopping
                    var_dump_best = {var.name: self.run(var) for var in self.variables}
                else:
                    patience -= 1
                if epoch > early_stopping and patience <= 0:
                    break

        train_time = perf_counter() - t

        if verbose and early_stop:
            print(f'converged after {epoch - patience} iterations')

        if val_nodes is not None:
            self.set_variables(var_dump_best)

            feed_dict = self.construct_feed_dict(x=x_val, y=y_val, dropout=0.)
            cost, acc = self.run([self.loss, self.accuracy], feed_dict=feed_dict)

            f1_micro, f1_macro = self.test_f1(x_val, y_val)

            return acc, train_time, f1_micro, f1_macro

    def construct_feed_dict(self, x, y, dropout=None):
        """Construct feed dictionary."""
        feed_dict = {
            self.x: x,
            self.y: y,
        }
        if dropout is not None:
            feed_dict[self.dropout] = dropout
        return feed_dict

    def set_variables(self, var_dump_best):
        with self.graph.as_default():
            self.assign_placeholders = {v.name: tf.placeholder(v.dtype, shape=v.get_shape()) for v in self.variables}
            self.assign_ops = {v.name: tf.assign(v, self.assign_placeholders[v.name]) for v in self.variables}
            self.run(list(self.assign_ops.values()), feed_dict={val: var_dump_best[key] for key, val in self.assign_placeholders.items()})

    def test(self, test_nodes):
        x = self.features[test_nodes]
        y = self.labels[test_nodes]
        return self.test_f1(x, y)

    def test_f1(self, x, y):
        feed_dict = self.construct_feed_dict(x=x, y=y)
        test_pred = self.run(self.outputs, feed_dict=feed_dict).argmax(1)
        test_real = y.argmax(1)
        return f1_score(test_real, test_pred, average='micro', labels=np.unique(test_pred)), f1_score(test_real, test_pred, average='macro', labels=np.unique(test_pred))

    def run(self, variables, feed_dict=None):
        return self.sess.run(variables, feed_dict=feed_dict)

    def reset_weight(self):
        self.run(tf.variables_initializer(self.variables))

    def predict(self, nodes):
        if np.isscalar(nodes):
            nodes = [nodes]
        x = self.features[nodes]
        y = self.labels[nodes]
        feed_dict = self.construct_feed_dict(x=x, y=y, dropout=0.)
        pred = self.run(self.outputs, feed_dict=feed_dict)
        return pred.squeeze()

    def get_weight(self):
        return self.run(self.variables)

    def close(self):
        self.sess.close()
