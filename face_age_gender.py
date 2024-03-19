# -*- coding: utf-8 -*-

#
#       顔検出 + 性別・年齢予測
#

#
#MIT License
#
#Copyright (c) 2017 Yusuke Uchida
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import cv2
import numpy as np
from wide_resnet import WideResNet
from pathlib import Path
import align.detect_face
import tensorflow as tf
from scipy import misc
from tensorflow.python.keras.utils.data_utils import get_file
import os.path as os
from PIL import Image
from omegaconf import OmegaConf
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import dlib
import traceback

def get_model(cfg):
    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    return model


# 性別・年齢を表記する関数
def draw_label(image, point, label, color, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), color, cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

# モデル取得
def get_loaded_model():    
    # モデルの設定
    if os.isdir("model") == False:
        pre_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
        modhash = '6d7f7b7ced093a8b3ef6399163da6ece'
        weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pre_model, cache_subdir="model",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
    else:
        weight_file = "model/EfficientNetB3_224_weights.11-3.44.hdf5"            

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)
    
    return model

def get_predict(model, faces):
    # 予測
    results = model.predict(faces)
    predicted_genders = results[0]
    ages = np.arange(0, 101).reshape(101, 1)
    predicted_ages = results[1].dot(ages).flatten()

    return predicted_ages, predicted_genders

def add_label(img, detected, predicted_ages):
    for i, d in enumerate(detected):
        if predicted_genders[i][0] < 0.5:
            color = (255, 128, 128)
            label = "{},{}".format(int(predicted_ages[i]), "Male")
        else:
            color = (128, 128, 255)
            label = "{},{}".format(int(predicted_ages[i]), "Female")
        draw_label(img, (d.left(), d.top()), label, color)
    

if __name__ == "__main__":
    #
    #   main 
    #        
    try:
        img = cv2.imread("test2.jpg") #入力画像
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = 224
        img_h, img_w, _ = np.shape(img)

        # for face detection
        detector = dlib.get_frontal_face_detector()
        
        # detect faces using dlib detector
        detected = detector(img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        
        model = get_loaded_model()

        #for face in range(len(faces)):        
        #    cv2.rectangle(img,(bb[face, 0], bb[face, 1]),(bb[face, 2], bb[face, 3]),(0,255,255),2)
        #    label = "{}, {}".format(int(Ages[face]), "Male" if Genders[face][0] < 0.5 else "Female")
        #    draw_label(img, (bb[face, 0], bb[face, 1]), label)
        margin = 1
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))
                
            predicted_ages, predicted_genders = get_predict(model, faces)
            add_label(img, detected, predicted_ages)

        # 出力画像の保存
        cv2.imwrite('static/images/output2.jpg', img)
    except:
        print(traceback.format_exc())
