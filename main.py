'''
監視器 + 平行 GPT-4 Vision + line 分析架構（修正版）
'''
import os
import json
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue
from gptvision import GPTVisionAnalyzer
from dotenv import load_dotenv

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import TextSendMessage
from linebot.models import MessageEvent, TextMessage, TextSendMessage

load_dotenv()  # 載入 .env
gpt_api_key = os.getenv("GPT4V_API_KEY")
line_channel_secret = os.getenv("Line_Channel_Secret")
line_channel_access_token = os.getenv("Line_Channel_Access_Token")
Your_User_ID = 'U5a68214ff2cf1d90c14112da96e42686'
app = Flask(__name__)
line_bot_api = LineBotApi(line_channel_access_token)
handler = WebhookHandler(line_channel_secret)
# ========== GPT 分析子程序（平行） ==========
def gpt_line_worker(frame_queue, line_token,target_user_id,api_key):
    '''從 Queue 取出影像 -> GPT 分析 -> LINE Push Message'''
    local_bot_api = LineBotApi(line_token)
    analyzer = GPTVisionAnalyzer(api_key=api_key)#子程序都能重新建立 instance，隔離性高：每個子程序有自己的 client，不會互相干擾
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        try:
            analysis = analyzer.analyze_frame(frame)
            # print(analysis)
            clean_json = analysis.replace("```json", "").replace("```", "").strip()
            info = json.loads(clean_json)
            print(f"🤖 GPT Analysis Result: {info['danger_level']}, {info['status']}, {info['reason']}")
            if info['danger_level'] == 'high' or info['danger_level'] == 'medium' or info['status'] !='ok':
                # 開新終端輸入：.\ngrok http 5000
                # 取得 ngrok 的網址+/callback，在https://developers.line.biz/console/channel/2008781464/messaging-api
                # 的wqebhook設定填入
                local_bot_api.push_message(
                    target_user_id,
                    TextSendMessage(text=f"🤖 GPT Analysis Result: {info['status']}, 危險等級: {info['danger_level']}\n原因: {info['reason']}")
                )
        except Exception as e:
            print(f"❌ GPT Worker 發生錯誤: {e}")


# ========== 主程式：動作偵測 ==========
def camera_worker(frame_queue):  # ← 修正：接收 Queue 作為參數
    '''接收影像找出差異的那一幀，並把該幀丟給 GPT Process'''
    # 開啟攝影機
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # 給攝影機一點啟動時間
    
    # 讀取第一幀作為參考
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    
    motion_active = False  # 紀錄目前是否處於動作中
    max_diff_value = 0  # 最大變化的數值
    stored_frame = None  # 保存最大變化的畫面
    
    while cap.isOpened():
        # 計算前後兩幀差異
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        
        # 找輪廓（輪廓越多或越大代表變化越大）
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False # 標記是否偵測到動作
        frame_diff_value = np.sum(diff)  # 用此判斷變化量
        
        for contour in contours:
            if cv2.contourArea(contour) < 15000:
                continue# 太小忽略，繼續下一個contour
            motion_detected = True #有動作
            # 標出移動區域
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # ========== 只在「動作剛開始」觸發 ==========
        if motion_detected:
            if not motion_active:
                print("⚠️ Motion Started!")
                motion_active = True
                max_diff_value = 0  # 重置最大差異
            
            # 更新最大變化幀
            if frame_diff_value > max_diff_value:
                max_diff_value = frame_diff_value
                stored_frame = frame1.copy()
        
        # ========== 動作結束：把「最大變化幀」丟給 GPT process ==========
        elif not motion_detected and motion_active:
            print("✅ Motion Stopped.")
            
            if stored_frame is not None:
                # 把幀送到 GPT Queue（先清空舊資料）
                if frame_queue.full():
                    frame_queue.get()  # 丟掉舊的
                frame_queue.put(stored_frame)  # 放入新的
                
                print("📤 已把最大變化幀送給 GPT Process")
            
            # 重置
            motion_active = False
            stored_frame = None
            max_diff_value = 0
        
        # 顯示畫面
        cv2.imshow("Security Monitor", frame1)
        
        # 更新前後畫面
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break
        
        # 按 q 結束
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
@app.route("/callback", methods=['POST'])
def callback():
    '''line路由'''
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'
if __name__ == "__main__":
    frame_queue = Queue(maxsize=1)
    # 啟動 GPT 分析子程序
    gpt_process = Process(
        target=gpt_line_worker, 
        args=(frame_queue,line_channel_access_token,Your_User_ID,gpt_api_key)
    )
    # 啟動攝影機偵測主程序
    cam_process = Process(
        target=camera_worker, 
        args=(frame_queue,)
    ) 
    gpt_process.start()
    cam_process.start()
    try:
        app.run(port=5000,use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        frame_queue.put(None)  # 給子程序一個結束信號
        gpt_process.join() 
        cam_process.terminate() # Camera 卡在 OpenCV 迴圈，直接強制關閉比較快
        cam_process.join()

        