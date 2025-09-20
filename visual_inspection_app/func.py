from PIL import Image
import io
import os
import timeit
import base64
import mlflow
from dotenv import load_dotenv
from openai import OpenAI

# .envファイルを読み込み
load_dotenv()

serving_endpoint_name = os.getenv("SERVING_ENDPOINT_NAME")
base_url = os.getenv("BASE_URL")
IMAGE_RESIZE = 256

# Databricks認証の設定確認とクライアント初期化
def initialize_databricks_client():
    """Databricksクライアントを初期化する"""
    try:
        # 必要な環境変数の確認
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        os.environ["DATABRICKS_HOST"] = databricks_host
        os.environ["DATABRICKS_TOKEN"] = databricks_token
        
        if not databricks_host or not databricks_token:
            import streamlit as st
            st.warning("DATABRICKS_HOSTまたはDATABRICKS_TOKENが設定されていません")
            st.info("デモモードで実行します（実際の推論は行われません）")
            return None
        
        client = mlflow.deployments.get_deploy_client("databricks")
        print("Databricksクライアントの初期化に成功しました")
        return client
    except Exception as e:
        print(f"Databricksクライアントの初期化に失敗しました: {e}")
        print("デモモードで実行します（実際の推論は行われません）")
        return None

def resize_image(img):
    width, height = img.size   # Get dimensions
    new_size = min(width, height)
    # Crop the center of the img
    img = img.crop(((width - new_size)/2, (height - new_size)/2, (width + new_size)/2, (height + new_size)/2))
    #Resize to the new resolution
    img = img.resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
    return img

def convert_PIL_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

# 入力をbase64に変換（サーバーレスモデルエンドポイントが受け取る形式）
def to_base64(byte):
    return base64.b64encode(byte).decode("ascii")

# Appが実行する関数
def predict(img, client=None):
    t_start = timeit.default_timer()
    img = resize_image(img)
    img_bytes = convert_PIL_to_bytes(img)
    
    # クライアントが初期化されていない場合はデモモードで実行
    if client is None:
        import streamlit as st
        st.info("デモモード: 実際の推論の代わりにサンプル結果を返します")
        demo_result = {
            "predictions": [
                {
                    "label": "normal",
                    "score": 0.85
                }
            ]
        }
        execution_time = round((timeit.default_timer() - t_start), 3)
        st.write(f"推論時間（エンドツーエンド）: {execution_time} s")
        return demo_result
    
    inferences = client.predict(
        endpoint = serving_endpoint_name, 
        inputs = {
            "inputs": {"data": to_base64(img_bytes)}
        })
    execution_time = round((timeit.default_timer() - t_start), 3)
    import streamlit as st
    st.write(f"推論時間（エンドツーエンド）: {execution_time} s")
    return inferences

# エージェント機能
def investigate_anomaly_with_agent(user_question=None):
    """異常検知結果をエージェントに送信して原因調査を実行"""
    try:
        # Databricks Token を取得
        databricks_token = os.getenv('DATABRICKS_TOKEN')
        if not databricks_token:
            return "エラー: DATABRICKS_TOKENが設定されていません"
        
        # OpenAI クライアントを初期化
        client = OpenAI(
            api_key = databricks_token,
            base_url = base_url
        )
        
        prompt = f"質問: {user_question}"

        # エージェントに問い合わせ
        response = client.responses.create(
            model="mas-631ab12a-endpoint",
            input=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.output[0].content[0].text
        
    except Exception as e:
        return f"エージェント実行エラー: {str(e)}"

# 画像とテキストの両方を使ったチャット機能
def chat_with_image_and_text(image=None, user_question=""):
    """画像とテキストを組み合わせたチャット機能
    
    Args:
        image: バイナリ形式の画像データ（bytes）
        user_question: ユーザーからの質問文字列
    """
    try:
        # Databricks Token を取得
        databricks_token = os.getenv('DATABRICKS_TOKEN')
        if not databricks_token:
            return "エラー: DATABRICKS_TOKENが設定されていません"
        
        # OpenAI クライアントを初期化
        client = OpenAI(
            api_key=databricks_token,
            base_url=base_url
        )
        
        # メッセージを構築
        content = [{"type": "text", "text": user_question}]
        
        # 画像が提供されている場合は追加
        if image is not None:
            # バイナリデータからBase64へ変換
            encoded_image = base64.b64encode(image).decode('utf-8')
            
            # # Base64データをテキストファイルにエクスポート
            # with open('encoded_image.txt', 'w') as f:
            #     f.write(encoded_image)
                
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })
        
        messages = [{"role": "user", "content": content}]
        
        # Claude Sonnet 4に問い合わせ
        response = client.chat.completions.create(
            model="databricks-claude-sonnet-4",
            messages=messages,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Claude チャットエラー: {str(e)}"