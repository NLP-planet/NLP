'''
@Project ：NLP 
@File    ：crf、bi-lstm-crf、idcnn-crf命名实体识别.py
@IDE     ：PyCharm 
@Author  ：Gaogz
@Date    ：2024/4/29 00:42 
@Desc    ：
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from collections import Counter
import tensorflow.keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Lambda, Conv1D, Dropout, concatenate, Input
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.datasets import conll2000
import tensorflow as tf
from keras.layers.normalization import BatchNormalization

EPOCHS = 10
EMBED_DIM = 200
BiRNN_UNITS = 200


def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics,
    reports per classs recall, precision and F1 score'''
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    '召回率（Recall）',
                                                    '精确率（Precision）',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')


# ------
# Data
# -----

# conll200 has two different targets, here will only use
# IBO like chunking as an example
train, test, voc = conll2000.load_data()
(train_x, _, train_y) = train
(test_x, _, test_y) = test
(vocab, _, class_labels) = voc


# --------------
# 1. Regular CRF
# --------------
def regular_crf(train_x, train_y, test_x, test_y):
    print('==== training CRF ====')

    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    # The default `crf_loss` for `learn_mode='join'` is negative log likelihood.
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])

    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]

    print('\n---- Result of CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)


# -------------
# 2. BiLSTM-CRF
# -------------

def bilstm_crf(train_x, train_y, test_x, test_y):
    print('==== training BiLSTM-CRF ====')

    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])

    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]

    print('\n---- Result of BiLSTM-CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)


# -------------
# 3. IDCNN-CRF
# -------------

class MaskedConv1D(Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None and self.padding == 'valid':
            mask = mask[:, self.kernel_size[0] // 2 * self.dilation_rate[0] * 2:]
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)


def IDCNN(input, cnn_filters=128, cnn_kernel_size=3, cnn_blocks=4, **kwargs):
    def _dilation_conv1d(dilation_rate):
        return MaskedConv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, padding="same",
                            dilation_rate=dilation_rate)

    def _idcnn_block():
        idcnn_1 = _dilation_conv1d(1)
        idcnn_2 = _dilation_conv1d(1)
        idcnn_3 = _dilation_conv1d(2)
        return [idcnn_1, idcnn_2, idcnn_3]

    input = BatchNormalization(name='normalization')(input)

    stack_idcnn_layers = []
    for layer_idx in range(cnn_blocks):
        idcnn_block = _idcnn_block()
        cnn = idcnn_block[0](input)
        cnn = Dropout(0.02)(cnn)
        cnn = idcnn_block[1](cnn)
        cnn = Dropout(0.02)(cnn)
        cnn = idcnn_block[2](cnn)
        cnn = Dropout(0.02)(cnn)
        stack_idcnn_layers.append(cnn)
    stack_idcnn = concatenate(stack_idcnn_layers, axis=-1)
    return stack_idcnn


def seq_padding(X, padding=0, max_len=100):
    if len(X.shape) == 2:
        return np.array([
            np.concatenate([[padding] * (max_len - len(x)), x]) if len(x) < max_len else x for x in X
        ])
    elif len(X.shape) == 3:
        return np.array([
            np.concatenate([[[padding]] * (max_len - len(x)), x]) if len(x) < max_len else x for x in X
        ])
    else:
        return X


def idcnn_crf(train_x, train_y, test_x, test_y):
    test_x = seq_padding(test_x, padding=0, max_len=train_x.shape[1])
    test_y = seq_padding(test_y, padding=-1, max_len=train_y.shape[1])

    print('==== training IDCNN-CRF ====')

    # build models
    input = Input(shape=(train_x.shape[-1],))
    emb = Embedding(len(vocab), EMBED_DIM, mask_zero=True)(input)
    idcnn = IDCNN(emb)
    crf_out = CRF(len(class_labels), sparse_target=True)(idcnn)
    model = Model(input, crf_out)
    model.summary()

    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])

    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]

    print('\n---- Result of IDCNN-CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)


def main():
    regular_crf(train_x, train_y, test_x, test_y)
    bilstm_crf(train_x, train_y, test_x, test_y)
    idcnn_crf(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()