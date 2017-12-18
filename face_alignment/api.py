from __future__ import print_function
import os
import glob
import dlib
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from enum import Enum
from skimage import io
from scipy.io import loadmat
from scipy.misc import imresize
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *
import json
import numpy as np
from numpy.random import shuffle, seed, randint, rand, uniform
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from itertools import chain
import time

class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """Initialize the face alignment pipeline

    Args:
        landmarks_type (``LandmarksType`` object): an enum defining the type of predicted points.
        network_size (``NetworkSize`` object): an enum defining the size of the network (for the 2D and 2.5D points).
        enable_cuda (bool, optional): If True, all the computations will be done on a CUDA-enabled GPU (recommended).
        enable_cudnn (bool, optional): If True, cudnn library will be used in the benchmark mode
        flip_input (bool, optional): Increase the network accuracy by doing a second forward passed with
                                    the flipped version of the image
        use_cnn_face_detector (bool, optional): If True, dlib's CNN based face detector is used even if CUDA
                                                is disabled.

    Example:
        >>> FaceAlignment(NetworkSize.2D, flip_input=False)
    """

    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 enable_cuda=True, enable_cudnn=True, flip_input=False,
                 use_cnn_face_detector=False, fold=0):
        self.enable_cuda = enable_cuda
        self.use_cnn_face_detector = use_cnn_face_detector
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if enable_cudnn and self.enable_cuda:
            torch.backends.cudnn.benchmark = True

        # Initialise the face detector
        if self.enable_cuda or self.use_cnn_face_detector:
            path_to_detector = os.path.join(
                base_path, "mmod_human_face_detector.dat")
            if not os.path.isfile(path_to_detector):
                print("Downloading the face detection CNN. Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat",
                    os.path.join(path_to_detector))

            self.face_detector = dlib.cnn_face_detection_model_v1(
                path_to_detector)

        else:
            self.face_detector = dlib.get_frontal_face_detector()

        # Initialise the face alignemnt networks
        self.face_alignemnt_net = FAN(int(network_size))
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(int(network_size)) + '.pth.tar'
        else:
            network_name = '3DFAN-' + str(int(network_size)) + '.pth.tar'
        fan_path = os.path.join(base_path, network_name)
        #fan_path = "/scratch/zhaosh/face_alignment/Models/retrain_2layer/retrain_2d_fold"+fold+ "_epoch9.pth.tar"

        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_path))
        print(fan_path)
        fan_weights = torch.load(
            fan_path,
            map_location=lambda storage,
            loc: storage)
        fan_dict = {k.replace('module.', ''): v for k,
                    v in fan_weights['state_dict'].items()}

        self.face_alignemnt_net.load_state_dict(fan_dict)

        if self.enable_cuda:
            self.face_alignemnt_net.cuda()
        self.face_alignemnt_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_model_path = os.path.join(base_path, 'depth.pth.tar')
            if not os.path.isfile(depth_model_path):
                print(
                    "Downloading the Face Alignment depth Network (FAN-D). Please wait...")

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/python-fan/depth.pth.tar",
                    os.path.join(depth_model_path))

            depth_weights = torch.load(
                depth_model_path,
                map_location=lambda storage,
                loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            if self.enable_cuda:
                self.depth_prediciton_net.cuda()
            self.depth_prediciton_net.eval()
        

    def detect_faces(self, image):
        """Run the dlib face detector over an image

        Args:
            image (``ndarray`` object or string): either the path to the image or an image previosly opened
            on which face detection will be performed.

        Returns:
            Returns a list of detected faces
        """
        return self.face_detector(image, 1)

### WaLaa!
    def get_face(self, image_name):
        path, image_name = os.path.split(image_name)
        if 'profile' in path:
            bbox = loadmat('/scratch/zhaosh/Menpo/Data/Regina_profile/profile_bbox.mat')
            bbox = bbox['profile_bbox']
        else:
            bbox = loadmat('/scratch/zhaosh/Menpo/Data/Regina/bbox/frontal_bbox.mat')
            bbox = bbox['frontal_bbox']
        for i, face in enumerate(bbox):
            if str(face['name'][0][0]) == image_name:
                return [dlib.rectangle(int(face['left']),int(face['top']),int(face['right']),int(face['bot']))]
        return []

    def crop_bbox(self, image, d):
        center = torch.FloatTensor(
                [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                (d.bottom() - d.top()) / 2.0])
        center[1] = center[1] - (d.bottom() - d.top()) * 0.1
        scale = (d.right() - d.left() + d.bottom() - d.top()) / 200.0
        inp = crop(image, center, scale)
        return inp

    def process_input(self, image, d):
        inp = self.crop_bbox(image,d)
        inp = torch.from_numpy(inp.transpose(
            (2, 0, 1))).float().div(255.0)#.unsqueeze_(0)
        return inp

    def raw_output(self, image):
        out = self.face_alignemnt_net(
            Variable(image, volatile=True))#[-1].data.cpu()
        return out

    def get_gts(self, image_name):
        pts = []
        with open(image_name[:-4]+'.ljson') as data_file:
            data = json.load(data_file)
            pts = np.array(data['landmarks']['points'])
            pts[:,[0,1]] = pts[:,[1,0]]
        pts = torch.Tensor(pts)
        return pts

    def crop_pts(self, pts, d, resolution=256.0):
        center = torch.FloatTensor(
                 [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                 (d.bottom() - d.top()) / 2.0])
        center[1] = center[1] - (d.bottom() - d.top()) * 0.1
        scale = (d.right() - d.left() + d.bottom() - d.top()) / 200.0
        ul = transform([1, 1], center, scale, resolution, True)
        br = transform([resolution, resolution], center, scale, resolution, True)
        
        pts = pts.numpy()-ul.numpy()
        pts[:,0] = pts[:,0]/(br[0] - ul[0])*resolution
        pts[:,1] = pts[:,1]/(br[1] - ul[1])*resolution
        return torch.Tensor(pts)

    def label_pts(self, pts, resolution=256):
        heatmaps = np.zeros((68, resolution, resolution))
        for i in range(68):
            if pts[i, 0] > 0:
                heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 1)
        return torch.from_numpy(heatmaps).float()

    def get_regina_dataset(self, train_path, resolution=64):
        train_set = []
        for image_name in train_path:
            d = self.get_face(image_name)[0]
            image = io.imread(image_name)
            image[:,:,0] = (image[:,:,0]*uniform(0.7,1.3)).clip(0,255)
            image[:,:,1] = (image[:,:,1]*uniform(0.7,1.3)).clip(0,255)
            image[:,:,2] = (image[:,:,2]*uniform(0.7,1.3)).clip(0,255)
            inp = self.process_input(image, d)
            label = self.get_gts(image_name)
            label = self.crop_pts(label,d,resolution)
            label = self.label_pts(label,resolution)
            train_set.append((inp, label))
        return train_set

    def data_loader(self, dataset, random=True, minibatch=10, bMirror=False, iRotate=0, bLowRes=False):
        def rot(x,max_deg):
            if rand() <= 0.4:
                deg = randint(max_deg*2+1)-max_deg
                x = (rotate(x[0],deg),rotate(x[1],deg))
            return x
        def mir(x):
            if rand() <= 0.5:
                x = (mirror(x[0]), mirror(x[1],True))
            return x
        def low_res(x):
            if rand() <= 0.2:
                (inp,label) = x
                inp = inp.numpy().transpose(1,2,0)
                inp = imresize(inp,size=(96,96))
                inp = imresize(inp,size=(256,256))
                inp = torch.from_numpy(inp.transpose(2,0,1)).float().div(255.0)
                x = (inp, label)
            return x

        if random:
            shuffle(dataset)
        if minibatch:
            batch_size = minibatch
        else:
            batch_size = len(dataset)

        if bLowRes:
            dataset = list(map(lambda x:low_res(x), dataset))
        if bMirror:
            dataset = list(map(lambda x:mir(x), dataset))
            #dataset = [(mirror(item[0]), mirror(item[1],True)) for item in dataset]
        if iRotate:
            dataset = list(map(lambda x:rot(x,iRotate), dataset))
            #dataset = [(rotate(item[0],deg), rotate(item[1],deg)) for item in dataset]
        data = [(torch.stack([item[0] for item in batch]),torch.stack([item[1] for item in batch])) for batch in zip(*[iter(dataset)]*batch_size)]

        return data

    def train(self, train_path, epoch, bMirror=False, iRotate=0, bLowRes=False):
        train_set = self.get_regina_dataset(train_path)

        self.face_alignemnt_net.train()
        
        end = time.time()
        for batch_idx, (inp, label) in enumerate(self.data_loader(train_set, random=True, bMirror=bMirror, iRotate=iRotate, bLowRes=bLowRes)):
            inp, label = Variable(inp), Variable(label)
            self.optimizer.zero_grad()
            out = self.face_alignemnt_net(inp)[-1]
            loss = F.mse_loss(out, label)
            loss.backward()
            self.optimizer.step()
            if (batch_idx+1) % 10 == 0:
                print('Train Epoch: {:3d} | batch: {:3d}/{:3d} | Time Spent: {:4.1f}min | Training Loss: {:.6f}'.format(
                    epoch, batch_idx+1, len(train_set)/10, (time.time()-end)/60, loss.data[0]))
                end = time.time()


    def validate(self, valid_path, epoch):
        valid_set = self.get_regina_dataset(valid_path)
        end = time.time()
        v_inp, v_label = self.data_loader(valid_set, random=False, minibatch=0)[-1] # validation set is the whole list
        v_out = self.face_alignemnt_net(Variable(v_inp, volatile=True))[-1]
        v_loss = F.mse_loss(v_out, Variable(v_label, volatile=True))
        print('Valid Epoch: {:3d} | Time Spent: {:4.1f}min | Validation Loss: {:.6f}'.format(
            epoch, (time.time()-end)/60, v_loss.data[0]))
        self.scheduler.step(v_loss.data[0])
        end = time.time()


    def train_FAN(self, fold=0):
        self.v_loss = 1000000
        for param in self.face_alignemnt_net.parameters():
            param.requires_grad = False
        for param in chain(#self.face_alignemnt_net.m3.parameters(),
                #self.face_alignemnt_net.top_m_3.parameters(),
                self.face_alignemnt_net.conv_last3.parameters(),
                self.face_alignemnt_net.l3.parameters(),
                self.face_alignemnt_net.bn_end3.parameters()):
            param.requires_grad = True
        #for param in self.face_alignemnt_net.l3.parameters():
        #    param.requires_grad = True
        #for param in self.face_alignemnt_net.bn_end3.parameters():
        #    param.requires_grad = True
        
        images_path = '/home/zhaosh/scratch/Menpo/Data/Regina/images/'
        sideview_path = '/home/zhaosh/scratch/Menpo/Data/Regina_profile/Annotated_Images/'
        subjects = sorted(list(set([[num for num in os.path.split(image_name)[1].split('_')[3:5] if num.isdigit()][0] for image_name in glob.glob(images_path+'*.png')])))
        side_only = sorted(list(set([[num for num in os.path.split(image_name)[1].split('_')[3:5] if num.isdigit()][0] for image_name in glob.glob(sideview_path+'*.png')]) - set(subjects)))
        side_groups = [3,3,3,3,3]
        sideSub = np.split(side_only, np.cumsum(side_groups)[:-1])
        #TODO: small resize/shift on bounding box?
        groups = [18,17,17,17,17] # should be [18, 17, 17, 17, 17]
        testSub = np.split(subjects, np.cumsum(groups)[:-1]) #17~18 tests
        trainSub = [list(set(subjects)-set(test)) for test in testSub] #70~71 train/valid
        validSub = [train[63:] for train in trainSub]
        trainSub = [train[:63] for train in trainSub]
        testSub = [list(test)+list(sideSub[i]) for i, test in enumerate(testSub)] # add subjects that are only in sideview

        trainset, validset, testset = trainSub[fold], validSub[fold], testSub[fold]
        
        trainset = [image_name for image_name in glob.glob(images_path+'*.png') for subject in trainset if 'patient_'+subject+'_' in image_name]
        validset = [image_name for image_name in glob.glob(images_path+'*.png') for subject in validset if 'patient_'+subject+'_' in image_name]
        testset = [image_name for image_name in glob.glob(images_path+'*.png')+glob.glob(sideview_path+'*.png') for subject in testset if 'patient_'+subject+'_' in image_name]

        self.optimizer = optim.RMSprop(chain(#self.face_alignemnt_net.m3.parameters(),
            #self.face_alignemnt_net.top_m_3.parameters(),
            self.face_alignemnt_net.conv_last3.parameters(),
            self.face_alignemnt_net.l3.parameters(),
            self.face_alignemnt_net.bn_end3.parameters()), lr=0.00025)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.2, patience=1)

        models_path = '/home/zhaosh/scratch/face_alignment/Models/'
        for epoch in range(10):
            self.train(trainset, epoch, True, 0, True)
            self.validate(validset, epoch)

            torch.save({
                'epoch': epoch+1,
                'state_dict': self.face_alignemnt_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                }, models_path+'retrain_2d_fold'+str(fold)+'_epoch'+str(epoch)+'.pth.tar')
        return self.process_list(testset)
   
    def load_FAN(self, fan_path):
        fan_weights = torch.load(
                fan_path,
                map_location=lambda storage,
                loc: storage)
        fan_dict = {k.replace('module.', ''): v for k,
                v in fan_weights['state_dict'].items()}
        self.face_alignemnt_net.load_state_dict(fan_dict)


    def test_FAN(self, fan_path, fold):
        images_path = '/home/zhaosh/scratch/Menpo/Data/Regina/images/'
        sideview_path = '/home/zhaosh/scratch/Menpo/Data/Regina_profile/Annotated_Images/'
        subjects = sorted(list(set([[num for num in os.path.split(image_name)[1].split('_')[3:5] if num.isdigit()][0] for image_name in glob.glob(images_path+'*.png')])))
        side_only = sorted(list(set([[num for num in os.path.split(image_name)[1].split('_')[3:5] if num.isdigit()][0] for image_name in glob.glob(sideview_path+'*.png')]) - set(subjects)))
        side_groups = [3,3,3,3,3]
        sideSub = np.split(side_only, np.cumsum(side_groups)[:-1])

        groups = [18,18,18,17,17] # should be [18, 17, 17, 17, 17]
        testSub = np.split(subjects, np.cumsum(groups)[:-1]) #17~18 tests
        testSub = [list(test)+list(sideSub[i]) for i, test in enumerate(testSub)] # add subjects that are only in sideview
        
        testset = testSub[fold]
        testset = [image_name for image_name in glob.glob(images_path+'*.png')+glob.glob(sideview_path+'*.png') for subject in testset if 'patient_'+subject+'_' in image_name]

        self.load_FAN(fan_path)
        return self.process_list(testset)
        


    def process_list(self, images_list):
        predictions = []
        for image_name in images_list:
            print (image_name)
            result = {'name':image_name,
                      'lmks':self.get_landmarks(image_name)}
            if result['lmks'] == None or len(result['lmks']) == 0:
                result['lmks'] = 0
            predictions.append(result)

        return predictions
###
    
    def get_features(self, input_image, all_faces=False):
        if isinstance(input_image, str):
            try:
                image = io.imread(input_image)
            except IOError:
                print("error opening file :: ", input_image)
                return None
        else:
            image = input_image
            
        detected_faces = self.get_face(input_image)
        if len(detected_faces) == 0:
            detected_faces = self.detect_faces(image)
        if len(detected_faces) > 0:
            landmarks = []
        for i, d in enumerate(detected_faces):
            if i > 1 and not all_faces:
                break
            if self.enable_cuda or self.use_cnn_face_detector:
                d = d.rect
            center = torch.FloatTensor(
                    [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                    (d.bottom() - d.top()) / 2.0])
            center[1] = center[1] - (d.bottom() - d.top()) * 0.1
            scale = (d.right() - d.left() + d.bottom() - d.top()) / 200.0
            if len(image.shape) == 2:
                image = np.stack((image,image,image),axis=-1)
            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                  (2, 0, 1))).float().div(255.0).unsqueeze_(0)

            out = [output.data.cpu() for output in self.face_alignemnt_net(Variable(inp, volatile=True))]
            return out



    def get_landmarks(self, input_image, all_faces=False):
        if isinstance(input_image, str):
            try:
                image = io.imread(input_image)
            except IOError:
                print("error opening file :: ", input_image)
                return None
        else:
            image = input_image

        detected_faces = self.get_face(input_image)
        if len(detected_faces) == 0:
            detected_faces = self.detect_faces(image)
        if len(detected_faces) > 0:
            landmarks = []
            for i, d in enumerate(detected_faces):
                if i > 1 and not all_faces:
                    break
                if self.enable_cuda or self.use_cnn_face_detector:
                    d = d.rect

                center = torch.FloatTensor(
                    [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                     (d.bottom() - d.top()) / 2.0])
                center[1] = center[1] - (d.bottom() - d.top()) * 0.1
                scale = (d.right() - d.left() + d.bottom() - d.top()) / 200.0
                
                if len(image.shape) == 2:
                    image = np.stack((image,image,image),axis=-1)

                inp = crop(image, center, scale)
                inp = torch.from_numpy(inp.transpose(
                    (2, 0, 1))).float().div(255.0).unsqueeze_(0)

                if self.enable_cuda:
                    inp = inp.cuda()

                out = self.face_alignemnt_net(
                    Variable(inp, volatile=True))[-1].data.cpu()
                if self.flip_input:
                    out += flip(self.face_alignemnt_net(Variable(flip(inp),
                                                                 volatile=True))[-1].data.cpu(), is_label=True)

                pts, pts_img = get_preds_fromhm(out, center, scale)
                pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

                if self.landmarks_type == LandmarksType._3D:
                    heatmaps = np.zeros((68, 256, 256))
                    for i in range(68):
                        if pts[i, 0] > 0:
                            heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
                    heatmaps = torch.from_numpy(
                        heatmaps).view(1, 68, 256, 256).float()
                    if self.enable_cuda:
                        heatmaps = heatmaps.cuda()
                    depth_pred = self.depth_prediciton_net(
                        Variable(
                            torch.cat(
                                (inp, heatmaps), 1), volatile=True)).data.cpu().view(
                        68, 1)
                    pts_img = torch.cat(
                        (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

                landmarks.append(pts_img.numpy())
        else:
            print("Warning: No faces were detected.")
            return None

        return landmarks

    def process_folder(self, path, all_faces=False):
        types = ('*.jpg', '*.png')
        images_list = []
        for files in types:
            images_list.extend(glob.glob(path+files))

        predictions = []
        for image_name in images_list:
            print (image_name)
            result = {'name':image_name,
                      'lmks':self.get_landmarks(image_name, all_faces)}
            if result['lmks'] == None or len(result['lmks']) == 0:
                result['lmks'] = 0
            predictions.append(result)

        return predictions



