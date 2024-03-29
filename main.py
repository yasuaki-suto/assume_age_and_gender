import cv2
from PIL import Image
from pathlib import Path

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys

import traceback

from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)

from linebot.exceptions import (
    InvalidSignatureError
)

from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FollowEvent,
    ImageMessage, ImageSendMessage, AudioMessage
)

from face_age_gender import get_loaded_model, get_predict, add_label
import numpy as np
import dlib


#環境変数取得
#LINE Developers->チャネル名->MessagingAPI設定
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('ENV_LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET       = os.getenv('ENV_LINE_CHANNEL_SECRET')
RENDER_URL = "https://assume-age-and-gender.onrender.com/"
#RENDER_URL = "http://localhost:8080/"
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app = Flask(__name__)

# 画像の保存
SRC_IMG_PATH = "static/images/{}.jpg"
def save_img(message_id, src_img_path):
    # message_idから画像のバイナリデータを取得
    message_content = line_bot_api.get_message_content(message_id)
    with open(src_img_path, "wb") as f:
        # バイナリを1024バイトずつ書き込む
        for chunk in message_content.iter_content():
            f.write(chunk)
            

@app.route("/")
def hello_world():
    return "hello world!"
    
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(FollowEvent)
def handle_follow(event):
   line_bot_api.reply_message(
       event.reply_token,
       TextSendMessage(text='友達追加ありがとう'))
       
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text))


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        message_id = event.message.id
        src_img_path = SRC_IMG_PATH.format(message_id)   # 保存する画像のパス
        save_img(message_id, src_img_path)   # 画像を一時保存する
        
        img = cv2.imread(src_img_path)
        img_size = 224
        img_h, img_w, _ = np.shape(img)
        '''
        
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
            add_label(img, detected, predicted_genders, predicted_ages)
        '''
        # 出力画像の保存
        cv2.imwrite('static/images/output.jpg', img)
    
    except:
        print(traceback.format_exc())
    #message_id = event.message.id
    #image_path = getImageLine(message_id)
    
    line_bot_api.reply_message(
        event.reply_token,[
        ImageSendMessage(
            original_content_url = RENDER_URL + "static/images/output.jpg",
            preview_image_url = RENDER_URL + "static/images/output.jpg"
        ),
        ]
    )
    
    # 一時保存していた画像を削除
    Path(SRC_IMG_PATH.format(message_id)).absolute().unlink()
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    #handle_image()
