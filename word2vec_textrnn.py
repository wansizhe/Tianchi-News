import os
import numpy as np
import pandas as pd
import warnings
from gensim.models import Word2Vec
from tqdm import tqdm
import random
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

from keras import backend as K
from keras.preprocessing import text, sequence
from keras import Model
from keras.layers import Conv1D, Embedding, Input, Bidirectional, CuDNNLSTM, Dense, Concatenate, Masking, LSTM, SpatialDropout1D
from keras.layers import BatchNormalization, Dropout, Activation
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D, GlobalAvgPool1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.utils import to_categorical
from keras_radam import RAdam
from keras_lookahead import Lookahead

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')


def fix_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


seed = 2020
fix_seed(seed)


df_train = pd.read_csv('input/train_set.csv', sep='\t')
df_test = pd.read_csv('input/test_a.csv', sep='\t')
df_data = df_train.append(df_test)
df_data = df_data.reset_index(drop=True)


max_words_num = None
seq_len = 2000
embedding_dim = 128
col = 'text'

print('Generate seqs')
os.makedirs('textrnn/seqs', exist_ok=True)
seq_path = 'textrnn/seqs/seqs_{}_{}.npy'.format(max_words_num, seq_len)
word_index_path = 'textrnn/seqs/word_index_{}_{}.npy'.format(max_words_num, seq_len)
if not os.path.exists(seq_path) or not os.path.exists(word_index_path):
    tokenizer = text.Tokenizer(
        num_words=max_words_num, lower=False, filters='')
    tokenizer.fit_on_texts(df_data[col].values.tolist())
    seqs = sequence.pad_sequences(tokenizer.texts_to_sequences(df_data[col].values.tolist()), maxlen=seq_len,
                                  padding='post', truncating='pre')
    word_index = tokenizer.word_index

    np.save(seq_path, seqs)
    np.save(word_index_path, word_index)

else:
    seqs = np.load(seq_path)
    word_index = np.load(word_index_path, allow_pickle=True).item()

embedding = np.zeros((len(word_index) + 1, embedding_dim))


os.makedirs('textrnn/model', exist_ok=True)
os.makedirs('textrnn/sub', exist_ok=True)
os.makedirs('textrnn/prob', exist_ok=True)


all_index = df_data[df_data['label'].notnull()].index.tolist()
test_index = df_data[df_data['label'].isnull()].index.tolist()


def build_model(emb, seq_len):
    emb_layer = Embedding(
        input_dim=emb.shape[0],
        output_dim=emb.shape[1],
        # weights=[emb],
        input_length=seq_len,
        # trainable=False
    )

    seq = Input(shape=(seq_len, ))
    seq_emb = emb_layer(seq)

    seq_emb = SpatialDropout1D(rate=0.2)(seq_emb)

    lstm = Bidirectional(CuDNNLSTM(200, return_sequences=True))(seq_emb)
    lstm_avg_pool = GlobalAveragePooling1D()(lstm)
    lstm_max_pool = GlobalMaxPooling1D()(lstm)
    x = Concatenate()([lstm_avg_pool, lstm_max_pool])

    x = Dropout(0.2)(Activation(activation='relu')
                     (BatchNormalization()(Dense(1024)(x))))
    out = Dense(14, activation='softmax')(x)

    model = Model(inputs=seq, outputs=out)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Lookahead(RAdam()), metrics=['accuracy'])

    return model


class Evaluator(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.best_val_f1 = 0.
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def evaluate(self):
        y_true = self.y_val
        y_pred = self.model.predict(self.x_val).argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = self.evaluate()
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
        logs['val_f1'] = val_f1
        print(f'val_f1: {val_f1:.5f}, best_val_f1: {self.best_val_f1:.5f}')


# train
bs = 256
monitor = 'val_f1'

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold_id, (train_index, val_index) in enumerate(kfold.split(all_index, df_data.iloc[all_index]['label'])):
    train_x = seqs[train_index]
    val_x = seqs[val_index]

    label = df_data['label'].values
    train_y = label[train_index]
    val_y = label[val_index]

    model_path = 'textrnn/model/lstm_{}.h5'.format(fold_id)
    checkpoint = ModelCheckpoint(model_path, monitor=monitor, verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)
    earlystopping = EarlyStopping(
        monitor=monitor, patience=5, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor, factor=0.5, patience=2, mode='max', verbose=1)

    model = build_model(embedding, seq_len)
    model.fit(train_x, train_y, batch_size=bs, epochs=30,
              validation_data=(val_x, val_y),
              callbacks=[Evaluator(validation_data=(val_x, val_y)), checkpoint, reduce_lr, earlystopping], verbose=1, shuffle=True)


# test
oof_pred = np.zeros((len(all_index), 14))
test_pred = np.zeros((len(test_index), 14))

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
for fold_id, (train_index, val_index) in enumerate(kfold.split(all_index, df_data.iloc[all_index]['label'])):
    model = build_model(embedding, seq_len)
    model_path = 'textrnn/model/lstm_{}.h5'.format(fold_id)
    model.load_weights(model_path)

    val_x = seqs[val_index]
    prob = model.predict(val_x, batch_size=bs, verbose=1)
    oof_pred[val_index] = prob

    test_x = seqs[test_index]
    prob = model.predict(test_x, batch_size=bs, verbose=1)
    test_pred += prob / 5


df_oof = df_data.loc[all_index][['label']]
df_oof['predict'] = np.argmax(oof_pred, axis=1)
f1score = f1_score(df_oof['label'], df_oof['predict'], average='macro')
print(f1score)

np.save('textrnn/prob/sub_5fold_lstm_{}.npy'.format(f1score), test_pred)
np.save('textrnn/prob/oof_5fold_lstm_{}.npy'.format(f1score), oof_pred)

sub = pd.DataFrame()
sub['label'] = np.argmax(test_pred, axis=1)
sub.to_csv('output/lstm_{}.csv'.format(f1score), index=False)
