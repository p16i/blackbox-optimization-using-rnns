import tensorflow as tf
import numpy as np
import benchmarkfunctions as fun
import utils
import sys
import time

def kernelTF(x1,x2,l = 0.3):
    return tf.exp(-1.0/l**2*tf.reduce_sum((tf.expand_dims(x1,axis=2) - tf.expand_dims(x2,axis=1))**2, axis = 3))

def GPTF(X,A,x, l = 0.3):
    k_xX = kernelTF(tf.expand_dims(x, axis = 1),X)
    return tf.squeeze(tf.matmul(k_xX,  A),axis=(2,))

def normalize(minv, maxv, y):
        return 2*(y-minv)/(maxv-minv)-1.0

def train_rnn_n2n(dim, n_steps = 20, learning_rate_init=0.001, learning_rate_final=0.0001, epochs=1000, n_hidden = 50, batch_size = 160, loss_function='WSUM', logger=sys.stdout, close_session=True):
    tf.set_random_seed(1)

    learning_rate_decay_rate = (learning_rate_final/learning_rate_init) ** (1.0 / (epochs-1) )

    # declare utils
    debug = lambda x : (print(x, file=logger), logger.flush())

    # declare loss function
    loss_dict = {
        "MIN" : lambda x : tf.reduce_mean(tf.reduce_min(x, axis = 0)),
        "SUM" : lambda x : tf.reduce_mean(tf.reduce_sum(x, axis = 0)),
        "WSUM" : lambda x : \
            tf.reduce_mean(tf.reduce_sum(tf.multiply(x, np.linspace(1/(n_steps+1),1, n_steps+1)), axis = 0)),
        "EI" : lambda x : tf.reduce_mean(tf.reduce_sum(x, axis = 0))
            - tf.reduce_mean(tf.reduce_sum([tf.reduce_min(x[:i+1],axis = 0) for i in range(n_steps)], axis = 0)),
        'WSUM_EXPO': lambda x: \
             tf.reduce_mean(tf.reduce_sum(tf.multiply(x, np.power(0.5,np.arange(1,n_steps+1)[::-1])), axis = 0))
    }

    # load data
    X_train, A_train, min_train, max_train = utils.loadData(dim, 'training')
    X_test, A_test, min_test, max_test = utils.loadData(dim, 'testing')

    n_gp_samples = X_train.shape[1]

    # define model
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, dim]))
    }

    biases = {
        'out': tf.Variable(tf.random_normal([dim]))
    }


    size = tf.placeholder(tf.int32,[], name="size")

    Xt = tf.placeholder(tf.float32, [None, n_gp_samples, dim], name="Xt")
    At = tf.placeholder(tf.float32, [None, n_gp_samples, 1], name="At")
    mint = tf.placeholder(tf.float32, [None, 1], name="mmint")
    maxt = tf.placeholder(tf.float32, [None, 1], name="mmaxt")

    x_0 = -0.0*tf.ones([size, dim])
    h_0 = tf.ones([size, n_hidden])
    c_0 = tf.ones([size, n_hidden])

    state = (c_0, h_0)
    x = x_0
    y = normalize(mint, maxt, GPTF(Xt,At,x))
    sample_points = [x]
    samples_y = [y]

    f_min = y
    f_sum = 0

    scope = 'rnn-cell-%d' % int(time.time())

    # No idea why this is necessary
    cell = tf.contrib.rnn.LSTMCell(num_units = n_hidden, reuse=None)
    cell(tf.concat([x, y], 1), state, scope=scope)
    cell = tf.contrib.rnn.LSTMCell(num_units = n_hidden, reuse=True)

    for i in range(n_steps):
        h, state = cell(tf.concat([x, y], 1), state, scope=scope)
        x = tf.tanh(tf.matmul(h, weights['out']) + biases['out'])
        sample_points.append(x)

        y = normalize(mint, maxt, GPTF(Xt,At,x))
        samples_y.append(y)

    f_min = tf.reduce_mean(tf.reduce_min(samples_y, axis = 0))
    loss = loss_dict[loss_function](samples_y)

    learning_rate_tf = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(learning_rate_tf).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_loss_list = []
    test_loss_list = []
    train_fmin_list = []
    test_fmin_list = []

    # Train the Network
    debug("------------------------------------------------------------------------------------")
    debug("%-30s: %d" % ("Function Dimension", dim) )
    debug("%-30s: %s" % ("RNN Scope", scope) )
    debug("%-30s: %d" % ("Number of Training Samples", len(X_train)) )
    debug("%-30s: %d" % ("Number of Test Samples", len(X_test)) )
    debug("%-30s: %s" % ("Loss", loss_function) )
    debug("%-30s: %d" % ("Batch size", batch_size) )
    debug("%-30s: %d" % ("Number of hidden Units", n_hidden) )
    debug("%-30s: %d" % ("Sequence length", n_steps) )
    debug("%-30s: %d" % ("Epochs",epochs) )
    debug("%-30s: %.5f" % ("Learning rate init", learning_rate_init) )
    debug("%-30s: %.5f" % ("Learning rate decay", learning_rate_decay_rate) )
    debug("%-30s: %.5f" % ("Learning rate final", learning_rate_final) )
    debug("------------------------------------------------------------------------------------")

    learning_rate = learning_rate_init
    for ep in range(epochs):
        learning_rate = learning_rate * learning_rate_decay_rate

        for batch in range(len(X_train)//batch_size):
            X_batch = X_train[batch*batch_size:(batch+1)*batch_size]
            A_batch = A_train[batch*batch_size:(batch+1)*batch_size]
            min_batch = min_train[batch*batch_size:(batch+1)*batch_size]
            max_batch = max_train[batch*batch_size:(batch+1)*batch_size]

            sess.run([train_step], feed_dict={Xt: X_batch, At: A_batch, mint: min_batch, maxt: max_batch, size: X_batch.shape[0], learning_rate_tf: learning_rate})

        train_loss, train_fmin = sess.run([loss, f_min], feed_dict=\
                                          {Xt: X_train, At: A_train, mint: min_train, maxt: max_train, size: len(X_train)})
        test_loss, test_fmin = sess.run([loss, f_min], feed_dict=\
                                          {Xt: X_test, At: A_test, mint: min_test, maxt: max_test, size:len(X_test)})

        train_loss_list += [train_loss]
        test_loss_list += [test_loss]
        train_fmin_list += [train_fmin]
        test_fmin_list += [test_fmin]

        if ep < 10 or ep % (epochs // 10) == 0 or ep == epochs-1:
            msg = "Ep: %4d | TrainLoss : %.3f | TrainMin: %.3f | TestLoss: %.3f | TestMin: %.3f" % (ep, train_loss, train_fmin, test_loss, test_fmin)
            debug(msg)

    debug('Last output: %s' % msg)
    if close_session:
        sess.close()
    else:
        print('Leave session open')
        return sess, (samples_y, Xt, At, mint, maxt, size)

if __name__ == "__main__":
    print("run as main")
    dim = 2
    f = open('something-%d.txt' %dim, 'w')
    train_rnn_n2n(dim, epochs=2, logger=f)


