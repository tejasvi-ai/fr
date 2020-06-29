import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile
from timeit import default_timer as timer

from sphereface.dataset import ImageDataset
from cp2tform import get_similarity_transform_for_cv2
import sphereface.net_sphere
import tensorflow as tf

import DeepFace
from deepface.basemodels import OpenFace, Facenet, FbDeepFace

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff, extra_stats=False):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    if extra_stats:
        t_p = 1.0*np.count_nonzero((y_true==y_predict) & (y_true==1))/len(y_true[y_true==1])
        f_p = 1.0*np.count_nonzero((y_true==y_predict) & (y_true!=1))/len(y_true[y_true!=1])
        print('Thresh: ', threshold, 't_p: ', t_p, 'f_p: ', f_p, 'accuracy: ', accuracy)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

import tqdm
class _TQDM(tqdm.tqdm):
    def __init__(self, *argv, **kwargs):
        kwargs['disable'] = True
        if kwargs.get('disable_override', 'def') != 'def':
            kwargs['disable'] = kwargs['disable_override']
        super().__init__(*argv, **kwargs)
tqdm.tqdm = _TQDM

parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='openface', type=str)
parser.add_argument('--lfw', default='lfw.zip', type=str)
parser.add_argument('--model','-m', default='sphereface/model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()

model_name = None
predicts = []
if args.net == 'deepface':
    net = FbDeepFace.loadModel()
    model_name = 'DeepFace'
elif args.net == 'facenet':
    net = Facenet.loadModel()
    model_name = 'Facenet'
elif args.net == 'openface':
    net = OpenFace.loadModel()
    model_name = 'OpenFace'
else:
    net = getattr(net_sphere,args.net)()
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    net.feature = True

zfile = zipfile.ZipFile(args.lfw)

landmark = {}
with open('sphereface/data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]

with open('sphereface/data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

time = timer()
error = 0
for i in range(6000):
    if not i % 10: print(f"\r{i/6000*100:.3f}% after {timer()-time:.3f}s", end='')
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

    if model_name:
        try:
            cosdistance = DeepFace.verify(f'lfw/{name1}', f'lfw/{name2}', model_name=model_name, model=net, enforce_detection=False)["distance"]
        except ZeroDivisionError:
            cosdistance = 1
            error += 1
    else:
        img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1),landmark[name1])
        img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),landmark[name2])

        imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
            imglist[i] = (imglist[i]-127.5)/128.0

        img = np.vstack(imglist)
        img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
        output = net(img)
        f = output.data
        f1,f2 = f[0],f[2]
        cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))

accuracy = []
thd = []
folds = KFold(n=6000, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.05)
predicts = np.array([line.strip('\n').split() for line in predicts])
# predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
for idx, (train, test) in enumerate(folds):
    print(f"Testing {idx+1} fold out of {len(folds)} folds.")
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test], extra_stats=True))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
