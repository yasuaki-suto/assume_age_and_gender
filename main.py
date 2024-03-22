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
    '''
    try:
        message_id = event.message.id
        src_img_path = SRC_IMG_PATH.format(message_id)   # 保存する画像のパス
        save_img(message_id, src_img_path)   # 画像を一時保存する
        
        img = cv2.imread(src_img_path)
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
            add_label(img, detected, predicted_genders, predicted_ages)

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
    '''
    message_id = event.message.id
    src_img_path = SRC_IMG_PATH.format(message_id)   # 保存する画像のパス
    save_img(message_id, src_img_path)   # 画像を一時保存する
    input_file = src_img_path
    
    content = line_bot_api.get_message_content(event.message.id)
    content_b = b""
    for c in content.iter_content():
        content_b = content_b + c
        
    '''
    #input_file = "C:\\Users\\sutou\\Downloads\\20240223-000517.jpg"
    input_file = "C:\\Users\\sutou\\Downloads\\101774.jpg" #手動撮影未加工
    #input_file = "C:\\Users\\sutou\\Downloads\\29399.jpg"
    #input_file = "C:\\Users\\sutou\\Downloads\\DSC_0083~2.JPG" #手動撮影画像切り抜き

    '''
    img = cv2.imread(input_file)

    #with io.open(input_file, 'rb') as image_file:
    #    content = image_file.read()
    credentials = service_account.Credentials.from_service_account_file('helical-mile-415213-e08c46a18701.json')
    client = vision.ImageAnnotatorClient(credentials=credentials)

    image = vision.Image(content=content_b)
    response = client.text_detection(image=image)

    bounds = get_document_bounds(response, FeatureType.BLOCK)
    img_block = draw_boxes(input_file, bounds)

    bounds = get_document_bounds(response, FeatureType.PARA)
    img_para = draw_boxes(input_file, bounds)

    bounds = get_document_bounds(response, FeatureType.WORD)
    img_word = draw_boxes(input_file, bounds)

    bounds = get_document_bounds(response, FeatureType.SYMBOL)
    img_symbol = draw_boxes(input_file, bounds)

    plt.figure(figsize=[20,20])
    plt.subplot(141);plt.imshow(img_block[:,:,::-1]);plt.title("img_block")
    plt.subplot(142);plt.imshow(img_para[:,:,::-1]);plt.title("img_para")
    plt.subplot(143);plt.imshow(img_word[:,:,::-1]);plt.title("img_word")
    plt.subplot(144);plt.imshow(img_symbol[:,:,::-1]);plt.title("img_symbol")
    plt.savefig("static/images/img1.png", format='png')

    lines = get_sorted_lines(response)
    all_text=''
    for line in lines:
      texts = [i[2] for i in line]  # i[0]:x座標 i[1]:y座標 i[2]:文字 i[3]:vertices(=左上、右上、左下、右下のxy座標を持つ辞書)
      texts = ''.join(texts)
      bounds = [i[3] for i in line]
      #print(texts)
      all_text = all_text+texts + '\n'
      for bound in bounds:
        p1 = (bounds[0].vertices[0].x, bounds[0].vertices[0].y)   # top left
        p2 = (bounds[-1].vertices[1].x, bounds[-1].vertices[1].y) # top right
        p3 = (bounds[-1].vertices[2].x, bounds[-1].vertices[2].y) # bottom right
        p4 = (bounds[0].vertices[3].x, bounds[0].vertices[3].y)   # bottom left
        cv2.line(img, p1, p2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(img, p2, p3, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(img, p3, p4, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(img, p4, p1, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    plt.figure(figsize=[10,10])
    plt.axis('off')
    plt.imshow(img[:,:,::-1]);plt.title("img_by_line")
    #buf = io.BytesIO()
    #plt.savefig(buf, format='png')
    #plt.show()
    #グラフ表示しない
    #plt.close()
    #tmpfile = buf.getvalue()
    #png = base64.encodebytes(buf.getvalue()).decode("utf-8")
    plt.savefig("static/images/img2.png", format='png')
    
    print(all_text)
    #print(png)
    '''
    try:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=all_text))
    except Exception as e:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="申し訳ありません。何らかのエラーが発生しました。\n %s" % traceback.format_exc()))
    '''
        
    '''
    content = line_bot_api.get_message_content(event.message.id)
    with open('./receive.jpg', 'w') as f:
        for c in content.iter_content():
            f.write(c)    
    '''
    #message_id = event.message.id
    #image_path = getImageLine(message_id)
    
    line_bot_api.reply_message(
        event.reply_token,[
        ImageSendMessage(
            original_content_url = RENDER_URL + "static/images/img1.png",
            preview_image_url = RENDER_URL + "static/images/img1.png"
        ),
        ImageSendMessage(
            original_content_url = RENDER_URL + "static/images/img2.png",
            preview_image_url = RENDER_URL + "static/images/img2.png"
        ),
        TextSendMessage(text=all_text)
        ]
    )
    
    # 一時保存していた画像を削除
    Path(SRC_IMG_PATH.format(message_id)).absolute().unlink()
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    #handle_image()
