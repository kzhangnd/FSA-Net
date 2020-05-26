from os import path, makedirs
import cv2
import sys
sys.path.append('..')

import argparse
import numpy as np
from math import cos, sin
# from moviepy.editor import *
from lib.FSANET_model import *
from mtcnn.mtcnn import MTCNN

from keras import backend as K
from keras.layers import Average
from keras.models import Model

def results_mtcnn(detected, input_img, faces, ad, img_size, img_w, img_h, model):

    if len(detected) > 0:
        for i, d in enumerate(detected):
            #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            if d['confidence'] > 0.95: # discuss
                x1, y1, w, h = d['box']

                x2 = x1+w
                y2 = y1+h

                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)

                faces[i, :, :, :] = cv2.resize(
                    input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                faces[i, :, :, :] = cv2.normalize(
                    faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                face = np.expand_dims(faces[i, :, :, :], axis=0)
                p_result = model.predict(face)
                return p_result[0][0], p_result[0][1], p_result[0][2]
            else:
                return 500, 500, 500

    return 404, 404, 404

def main(source, destination):

    K.set_learning_phase(0)  # make sure its testing mode
    # face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    detector = MTCNN()

    # Parameters
    img_size = 64
    lambda_local = 1
    ad = 0.6
    detected = ''  # make this not local variabl
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    # load model and weights
    model1 = FSA_net_Capsule(image_size, num_classes,
                             stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    print('Loading models ...')

    weight_file1 = '../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')

    weight_file2 = '../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    print('Load image list ...')
    if path.isfile(source):
        source_list = np.sort(np.loadtxt(source, dtype=np.str))
    else:
        sys.exit('Unable to load image list')

    print('Check destination ...')
    if not path.exists(destination):
        makedirs(destination)
        print ("Make directory {}!".format(destination))
    save_file = path.join(destination, 'pose.txt')

    print('Start detecting pose ...')

    result = []
    for image_path in source_list:
        # get video frame
        input_img = cv2.imread(image_path)

        img_h, img_w, _ = np.shape(input_img)

        # detect faces using LBP detector
        #gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        # detected = face_cascade.detectMultiScale(gray_img, 1.1)
        detected = detector.detect_faces(input_img)

        faces = np.empty((len(detected), img_size, img_size, 3))

        yaw, pitch, roll = results_mtcnn(detected, input_img, faces, ad, img_size, img_w, img_h, model)
        result.append([image_path, yaw, pitch, roll])

    print('Save result file ...')
    np.savetxt(save_file, np.asarray(result), delimiter=' ', fmt='%s')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Head Pose Estimation for Image')
    parser.add_argument('--source', '-s', help='image source')
    parser.add_argument('--dest', '-d', help='Folder to save the estimations')

    args = parser.parse_args()
    main(args.source, args.dest)