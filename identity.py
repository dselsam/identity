import argparse
import numpy as np
import tensorflow as tf
from mlp import MLP
from util import repeat_end

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', action='store', dest='n_epochs', type=int, default=100000)
parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=128)
parser.add_argument('--n_inputs', action='store', dest='n_inputs', type=int, default=1)
parser.add_argument('--n_nodes_per_layer', action='store', dest='n_nodes_per_layer', type=int, default=64)
parser.add_argument('--n_layers', action='store', dest='n_layers', type=int, default=2)
parser.add_argument('--mlp_transfer_fn', action='store', dest='mlp_transfer_fn', type=str, default="relu")
parser.add_argument('--init_range', action='store', dest='init_range', type=float, default=0.2)
parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.0001)
parser.add_argument('--res_net', action='store', dest='res_net', type=int, default=0)
parser.add_argument('--train_range', action='store', dest='train_range', type=float, default=1.0)
parser.add_argument('--test_scale', action='store', dest='test_scale', type=float, default=10.0)
parser.add_argument('--print_freq', action='store', dest='print_freq', type=int, default=1000)
parser.add_argument('--l2_weight', action='store', dest='l2_weight', type=float, default=0.0)
parser.add_argument('--save_key', action='store', dest='save_key', type=str, default="default")
opts = parser.parse_args()

sess = tf.Session()

net = MLP(opts, opts.n_inputs, repeat_end(opts.n_nodes_per_layer, opts.n_layers, opts.n_inputs), name="net")

inputs  = tf.placeholder(tf.float32, shape=[None, opts.n_inputs], name='inputs')
targets = tf.placeholder(tf.float32, shape=[None, opts.n_inputs], name='outputs')

outputs = net.forward(inputs)

predict_loss = tf.reduce_mean(tf.square(outputs - targets))
l2_loss = tf.zeros([])
for var in tf.trainable_variables(): l2_loss += tf.nn.l2_loss(var)
loss = predict_loss + opts.l2_weight * l2_loss
update  = tf.train.AdamOptimizer(learning_rate=opts.learning_rate).minimize(loss)

tf.global_variables_initializer().run(session=sess)

for epoch in range(opts.n_epochs):
    x_train = np.random.uniform(low=-opts.train_range,
                                high=opts.train_range, size=(opts.batch_size, opts.n_inputs))
    x_test = np.random.uniform(low=-opts.test_scale * opts.train_range,
                               high=opts.test_scale * opts.train_range, size=(opts.batch_size, opts.n_inputs))
    train_loss, _    = sess.run((loss, update), feed_dict={ inputs : x_train, targets : x_train })
    test_loss        = sess.run(loss, feed_dict={ inputs : x_test, targets : x_test })
    if epoch % opts.print_freq == 0:
        print("[%5d] %.12f %.12f" % (epoch, train_loss, test_loss))


from pylab import *
import matplotlib.pyplot as plt

N_TICKS = 1000

plt.figure()
x_1  = np.reshape(np.linspace(-opts.train_range, opts.train_range, num=1000), (N_TICKS, 1))
y_net   = plt.plot(x_1, sess.run(outputs, feed_dict={inputs:x_1}))[0]
y_truth = plt.plot(x_1, x_1)[0]
plt.title('Learned function on training set')
legend([y_net, y_truth], ['Network', 'Identity'])

plt.savefig("figures/train.png")

plt.figure()
x_10 = np.reshape(np.linspace(-opts.test_scale * opts.train_range,
                              opts.test_scale * opts.train_range, num=1000), (N_TICKS, 1))
y_net   = plt.plot(x_10, sess.run(outputs, feed_dict={inputs:x_10}))[0]
y_truth = plt.plot(x_10, x_10)[0]
plt.title('Learned function on test set')
legend([y_net, y_truth], ['Network', 'Identity'])
plt.savefig("figures/test_%s.png" % opts.save_key)
