# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
from __future__ import print_function
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from lstm import lstm_unroll

from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random

import multiprocessing
import glob

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    buf = ""
    max_len = random.randint(3,4)
    for i in range(max_len):
        buf += str(random.randint(0,9))
        #buf += str(random.choice([str(i) for i in range(0, 9)]+[chr(i) for i in range(65, 91)]))
    return buf

def get_label(buf):
    ret = np.zeros(4)
    for i in range(len(buf)):
        ret[i] = 1 + int(buf[i])
    if len(buf) == 3:
        ret[3] = 0
    return ret

captcha = ImageCaptcha(fonts=['/Users/lonica/Downloads/Xerox.ttf'])
def gen_captcha():
    num = gen_rand()
    img = captcha.generate(num)
    img = np.fromstring(img.getvalue(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (80, 30))
    img = img.transpose(1, 0)
    img = img.reshape((80 * 30))
    img = np.multiply(img, 1/255.0)
    return num, img

class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, init_states):
        super(OCRIter, self).__init__()
        # you can get this font from http://font.ubuntu.com/
        self.captcha = ImageCaptcha(fonts=['/Users/lonica/Downloads/Xerox.ttf'])
        self.batch_size = batch_size
        self.count = count
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.provide_data = [('data', (batch_size, 2400))] + init_states
        self.provide_label = [('label', (self.batch_size, 4))]
        # self.pool = multiprocessing.Pool(processes=4)

    def __iter__(self):
        print('iter')
        init_state_names = [x[0] for x in self.init_states]
        f = glob.glob("samples/*.png")
        np.random.shuffle(f)
        for k in range(self.count):
            data = []
            label = []
            # result = []
            for i in range(self.batch_size):
                # result.append(self.pool.apply_async(gen_captcha))
                # num, img = gen_captcha()
                img = cv2.imread(f[k*BATCH_SIZE+i], cv2.IMREAD_GRAYSCALE)
                img = img.transpose(1, 0)
                img = img.reshape((80 * 30))
                img = np.multiply(img, 1 / 255.0)
                data.append(img)
                label.append(get_label(f[k*BATCH_SIZE+i].split(".")[0].split("_")[1]))
            # self.pool.close()
            # self.pool.join()
            # self.pool = multiprocessing.Pool(processes=4)
            #print(result[0].get())
            # data = [res.get(1)[1] for res in result]
            # label = [get_label(res.get(1)[0]) for res in result]
            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']


            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch



    def reset(self):
        pass

BATCH_SIZE = 32
SEQ_LENGTH = 80

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

def LCS(p,l):
    # Dynamic Programming Finding LCS
    if len(p) == 0:
        return 0
    P = np.array(list(p)).reshape((1, len(p)))
    L = np.array(list(l)).reshape((len(l), 1))
    M = np.int32(P == L)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            up = 0 if i == 0 else M[i-1,j]
            left = 0 if j == 0 else M[i,j-1]
            M[i,j] = max(up, left, M[i,j] if (i == 0 or j == 0) else M[i,j] + M[i-1,j-1])
    return M.max()


def Accuracy_LCS(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        hit += LCS(p,l) * 1.0 / len(l)
        total += 1.0
    return hit / total

if __name__ == '__main__':
    num_hidden = 100
    num_lstm_layer = 2

    num_epoch = 10
    learning_rate = 0.001
    momentum = 0.9
    num_label = 4

    contexts = [mx.context.cpu(0)]

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len,
                           num_hidden=num_hidden,
                           num_label = num_label)

    init_c = [('l%d_init_c'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = OCRIter(10000, BATCH_SIZE, num_label, init_states)
    data_val = OCRIter(1000, BATCH_SIZE, num_label, init_states)

    symbol = sym_gen(SEQ_LENGTH)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

<<<<<<< HEAD
    print('begin fit')
    prefix = "ocr"
=======
    prefix = 'ocr'
>>>>>>> e0f0bf0491ec4c0cfb22bf4ab92f06833f504699
    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy),
              # Use the following eval_metric if your num_label >= 10, or varies in a wide range
              # eval_metric = mx.metric.np(Accuracy_LCS),
              batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 50),
              epoch_end_callback = mx.callback.do_checkpoint(prefix, 1))

    model.save(prefix)
