import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

torch.backends.cudnn.bencmark = True

import os, sys, cv2, random, datetime
import argparse
import numpy as np
import zipfile
from timeit import default_timer as timer
from PIL import Image
from itertools import cycle
from glob import glob

from sphereface.dataset import ImageDataset
from cp2tform import get_similarity_transform_for_cv2
import sphereface.net_sphere as net_sphere
import tensorflow as tf

import DeepFace
from deepface.basemodels import OpenFace, Facenet, FbDeepFace

from random import shuffle


def alignment(src_img, src_pts, ref_pts=None):
    if not ref_pts:
        ref_pts = [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def youtube_alignment(src_img, src_pts, ref_pts=None):
    ref_pts = [
        [
            20.26656656,
            21.1005582,
            23.15464843,
            25.46972318,
            29.05898675,
            34.86671672,
            41.19116015,
            49.27502122,
            61.74502187,
            73.90835346,
            81.88933864,
            87.98509826,
            93.49918391,
            96.46479402,
            98.12920285,
            99.55467781,
            99.78380231,
            28.94963113,
            33.42872299,
            39.126167,
            44.41617157,
            49.28512437,
            68.8620487,
            73.5006039,
            78.93437031,
            84.90486061,
            89.67531175,
            59.40583992,
            59.54269766,
            59.84309917,
            60.1364497,
            53.75096298,
            56.50350917,
            60.29072273,
            63.89072599,
            66.65487693,
            36.54524385,
            40.13284259,
            44.90025788,
            49.2772573,
            45.34930143,
            40.29987922,
            69.73020174,
            74.12600379,
            78.90663968,
            82.73828099,
            78.84796305,
            73.92137821,
            46.73384148,
            51.36040021,
            57.27761637,
            60.43926683,
            63.4124339,
            69.27240974,
            73.8591924,
            69.22346086,
            65.26062545,
            60.71934778,
            56.12522034,
            51.72976105,
            47.6821832,
            56.47155122,
            60.43206894,
            64.47484821,
            72.86059607,
            64.40213488,
            60.55900307,
            56.48243781,
        ],
        [
            91.05280081,
            101.45836326,
            110.99072926,
            120.13964876,
            130.20327088,
            138.71861331,
            144.265016,
            149.29437879,
            152.16733042,
            148.47331396,
            143.30732193,
            137.42258602,
            128.92353268,
            118.55314357,
            109.35976366,
            99.77371548,
            89.20459294,
            80.65391395,
            77.34068355,
            76.34729386,
            76.93949533,
            78.26072338,
            77.87830515,
            76.48570216,
            75.85563426,
            76.78656068,
            79.80250702,
            89.62696677,
            96.72494614,
            103.44207417,
            109.15923484,
            113.44321669,
            114.22661096,
            114.9510348,
            114.02061435,
            113.13992623,
            89.1212052,
            87.71696481,
            87.58100477,
            89.5711138,
            90.96324345,
            91.07274597,
            89.22621923,
            87.21813998,
            87.04104916,
            88.45322191,
            90.45472351,
            90.5562447,
            126.66906052,
            123.65273879,
            121.7633675,
            122.33257165,
            121.6516942,
            123.41956649,
            126.32441079,
            130.5189985,
            132.93512111,
            133.50430894,
            133.18288503,
            130.89510022,
            126.49934713,
            125.28314944,
            125.17325521,
            125.28804596,
            126.15921852,
            128.14400666,
            128.6414115,
            128.34216883,
        ],
    ]
    return alignment(src_img, src_pts, ref_pts)


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i * n // n_folds : (i + 1) * n // n_folds]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff, extra_stats=True):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    if extra_stats:
        try:
            t_p = (
                1.0
                * np.count_nonzero((y_true == y_predict) & (y_true == 1))
                / len(y_true[y_true == 1])
            )
        except:
            t_p = "NaN"
        try:
            f_p = 1 - 1.0 * np.count_nonzero(
                (y_true == y_predict) & (y_true != 1)
            ) / len(y_true[y_true != 1])
        except:
            f_p = "NaN"
        print("Thresh: ", threshold, "t_p: ", t_p, "f_p: ", f_p, "accuracy: ", accuracy)
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
        kwargs["disable"] = True
        if kwargs.get("disable_override", "def") != "def":
            kwargs["disable"] = kwargs["disable_override"]
        super().__init__(*argv, **kwargs)


tqdm.tqdm = _TQDM

parser = argparse.ArgumentParser(description="PyTorch sphereface lfw")
parser.add_argument("--net", "-n", default="deepface", type=str)
parser.add_argument("--lfw", default="lfw.zip", type=str)
parser.add_argument("--flickr", default="flickr", type=str)
parser.add_argument("--yuoutbe", default="youv1", type=str)
parser.add_argument("--data", default="youtube", type=str)
parser.add_argument(
    "--model", "-m", default="sphereface/model/sphere20a_20171020.pth", type=str
)
args = parser.parse_args()

model_name = None
predicts = []
if args.net == "deepface":
    net = FbDeepFace.loadModel()
    model_name = "DeepFace"
elif args.net == "facenet":
    net = Facenet.loadModel()
    model_name = "Facenet"
elif args.net == "openface":
    net = OpenFace.loadModel()
    model_name = "OpenFace"
else:
    net = getattr(net_sphere, "sphere20a")()
    net.load_state_dict(torch.load(args.model))
    net.cuda()
    net.eval()
    net.feature = True

zfile = zipfile.ZipFile(args.lfw)

landmark = {}
with open("sphereface/data/lfw_landmark.txt") as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace("\n", "").split("\t")
    landmark[l[0]] = [int(k) for k in l[1:]]

with open("sphereface/data/pairs.txt") as f:
    pairs_lines = f.readlines()[1:]

same_name_list = []
diff_name_list = []
for i in range(6000):
    p = pairs_lines[i].replace("\n", "").split("\t")

    if 3 == len(p):
        sameflag = 1
        name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
        name2 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[2]))
    if 4 == len(p):
        sameflag = 0
        name1 = p[0] + "/" + p[0] + "_" + "{:04}.jpg".format(int(p[1]))
        name2 = p[2] + "/" + p[2] + "_" + "{:04}.jpg".format(int(p[3]))
    if sameflag:
        same_name_list.append((name1, name2, 1))
    else:
        diff_name_list.append((name1, name2, 0))
random.shuffle(same_name_list)
random.shuffle(diff_name_list)
name_list = diff_name_list + same_name_list

if args.data == "flickr":
    neg_list = glob(args.flickr + "/**/*.png", recursive=True)
    random.shuffle(neg_list)
    neg_list = cycle(neg_list)
elif args.data == "youtube":
    pass

time = timer()
error = 0
for i in range(len(name_list)):
    if not i % 10:
        print(f"\r{i/6000*100:.2f}% after {timer()-time:.2f}s", end="")
    sameflag = name_list[i][2]
    if sameflag:
        name1 = name_list[i][0]
    else:
        name1 = name_list[i // 100 * 100][0]
    name2 = name_list[i][1]
    if model_name:
        if not sameflag and args.data == "flickr":
            name1 = next(neg_list)
        else:
            name = "lfw/" + name
        try:
            cosdistance = (
                1
                - DeepFace.verify(
                    name1,
                    f"lfw/{name2}",
                    model_name=model_name,
                    model=net,
                    enforce_detection=False,
                )["distance"]
            )
        except ZeroDivisionError:
            cosdistance = 0.5
            error += 1
    else:
        img1 = alignment(
            cv2.imdecode(np.frombuffer(zfile.read(name1), np.uint8), 1), landmark[name1]
        )
        if not sameflag and args.data == "flickr":
            img2 = cv2.imread(next(neg_list))[8:120, 16:112, :]
        else:
            img2 = alignment(
                cv2.imdecode(np.frombuffer(zfile.read(name2), np.uint8), 1),
                landmark[name2],
            )

        imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
        for i in range(len(imglist)):
            imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            imglist[i] = (imglist[i] - 127.5) / 128.0

        img = np.vstack(imglist)
        img = Variable(torch.from_numpy(img).float(), volatile=True).cuda()
        output = net(img)
        f = output.data
        f1, f2 = f[0], f[2]
        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    predicts.append("{}\t{}\t{}\t{}\n".format(name1, name2, cosdistance, sameflag))

print("\n")
accuracy = []
thd = []
folds = KFold(n=6000, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.05)
predicts = np.array([line.strip("\n").split() for line in predicts])
# predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
for idx, (train, test) in enumerate(folds):
    print(f"Testing {idx+1} fold out of {len(folds)} folds.")
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test], extra_stats=True))
    thd.append(best_thresh)
print(
    "LFWACC={:.4f} std={:.4f} thd={:.4f}".format(
        np.mean(accuracy), np.std(accuracy), np.mean(thd)
    )
)
thd
