'''
analyze the given image and provide insights based on its content.
'''
import cv2
import base64
from openai import OpenAI, OpenAIError
class GPTVisionAnalyzer:
    def __init__(self, api_key):
        """
        初始化 GPT Vision 分析器
        
        參數:
            api_key: OpenAI API 金鑰
        """
        self.client = OpenAI(api_key=api_key)
    def encode_image(self, frame):
        """
        將 OpenCV 影像轉換為 base64 編碼
        
        參數:
            frame: OpenCV 讀取的影像 (numpy array)
        
        返回:
            base64 編碼的字串
        """
        # 將影像編碼為 JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        # 轉換為 base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    def analyze_frame(self, frame):
        """
        使用 GPT-4 Vision 分析影像
        
        參數:
            frame: OpenCV 影像
            prompt: 自訂提示詞（可選）
        
        返回:
            分析結果字串
        """
        # 預設提示詞
        prompt = (
            '''你會收到兩張圖片：
1. 第一張是「原始對照圖」，代表物體原本應該長什麼樣子。
2. 第二張是「目前畫面」，代表要檢查的現況。
請比較兩張圖中的物體（杯子）是否一致，並根據以下規則回覆：
- 如果第二張圖中杯子完全不見，只剩白色底，回覆 "stolen"
- 如果第二張圖中仍然是原本的杯子且未被更換，回覆 "ok"
- 如果第二張圖中出現了別的杯子或被替換過，回覆 "replaced"
只輸出以上三種字串之一，不要輸出任何其他內容。
'''
        )
        #對照圖
        with open("D:/my program/LLM/original.jpg", "rb") as f:
            original_image_data = f.read()
            base64_original = base64.b64encode(original_image_data).decode('utf-8')
        # 編碼影像
        base64_image = self.encode_image(frame)
        try:
            resp = self.client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_original}"   # 對照圖
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"  # 即時畫面
                            },
                            {
                                "type": "input_text",
                                "text": prompt
                            }
                        ],
                    }
                ]
            )
            return resp.output_text.strip()
        except OpenAIError as e:
            return f"發生錯誤:{e.status_code} {e}"
            