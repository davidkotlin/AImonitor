'''
監視器 + 平行 GPT-4 Vision 分析架構（修正版）
'''
import os
import cv2
import time
import numpy as np
from multiprocessing import Process, Queue
from gptvision import GPTVisionAnalyzer
from dotenv import load_dotenv
# ========== GPT 分析子程序（平行） ==========
def gpt_worker(frame_queue):
    '''從 Queue 取出影像並送給 GPT-4 Vision 分析'''
    load_dotenv()  # 載入 .env
    api_key = os.getenv("GPT4V_API_KEY")
    analyzer = GPTVisionAnalyzer(api_key=api_key)#子程序都能重新建立 instance，隔離性高：每個子程序有自己的 client，不會互相干擾
    while True:
        frame = frame_queue.get()
        try:
            analysis = analyzer.analyze_frame(frame)
            print(f"🤖 GPT Analysis Result: {analysis}")
        except Exception as e:
            print(f"❌ GPT Worker 發生錯誤: {e}")


# ========== 主程式：動作偵測 ==========
def main(frame_queue):  # ← 修正：接收 Queue 作為參數
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

if __name__ == "__main__":
    frame_queue = Queue(maxsize=1)
    gpt_process = Process(target=gpt_worker, args=(frame_queue,))
    gpt_process.start()
    
    main(frame_queue)  #傳入 Queue
    
    gpt_process.terminate()