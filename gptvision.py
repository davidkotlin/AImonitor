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
請比較兩張圖中的物體是否一致，並根據以下規則回覆json：
- **stolen**: 如果原本的物品消失了，且原本的位置變空了（只剩下背景）。
- **replaced**: 如果原本的物品消失了，但在該位置出現了「完全不同的物體」（例如不同的杯子、飲料罐、水果等）。
- **ok**: 如果原本的物品還在原本的位置（即使稍微被移動，只要確認是同一個杯子）。
以下為json格式範例：
{
  "status": "stolen" | "replaced" | "ok",
  "new_object_description": "如果不見或正常填 null，如果是被替換，請描述新物體",
  "danger_level": "low" | "medium" | "high",
  "reason": "簡短說明判斷理由"
}
注意：
1.如果在第二張圖中看到人類的手或身體的一部分，請判斷物體是否還在。如果手拿著物品懸空，請視為「stolen」（被拿走中）。
2.如果ok的情況下，請進一步檢查物體是否破損，依照破損程度回應danger_level
3.如果畫面中有「人」或「手」，在reason加以描述是否有戴手套、相關特徵、是否持有其他工具等

'''
        )
        #對照圖
        with open("wallet.jpg", "rb") as f:
        # with open("original.jpg", "rb") as f:
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
            