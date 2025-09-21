import streamlit as st
import pandas as pd
from PIL import Image
import io
import json
from func import predict, initialize_databricks_client, investigate_anomaly_with_agent, chat_with_image_and_text

# ページ設定
st.set_page_config(page_title="外観検査アプリ", layout="wide")

# クライアントを取得
client = initialize_databricks_client()

# チャット履歴の初期化
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_image_bytes' not in st.session_state:
    st.session_state.current_image_bytes = None

# メインタイトル
st.title("外観検査アプリ")

# サイドバーでページ選択
st.sidebar.title("ページ選択")
page = st.sidebar.selectbox(
    "機能を選択してください:",
    ["🔍 画像解析", "💬 AIエージェントチャット"],
    help="使用したい機能を選択してください"
)

# テキストサイズ調整機能
st.sidebar.header("表示設定")
text_size = st.sidebar.slider(
    "テキストサイズ",
    min_value=12,
    max_value=36,
    value=16,
    step=1,
    help="アプリ全体のテキストサイズを調整できます"
)

# 動的CSSスタイリング
st.html(f"""
<style>
/* 通常のテキスト要素のみサイズ調整（ヘッダー類は除外） */

/* ボタンのテキストサイズ */
.stButton > button {{
    font-size: {text_size}px !important;
}}

/* セレクトボックスのテキストサイズ */
.stSelectbox > div > div {{
    font-size: {text_size}px !important;
}}

/* テキストエリアのサイズ */
.stTextArea > div > div > textarea {{
    font-size: {text_size}px !important;
}}

/* チャットメッセージのテキストサイズ */
[data-testid="chatMessage"] {{
    font-size: {text_size}px !important;
}}

/* チャット入力のテキストサイズ */
.stChatInput > div > div > div > div > div > textarea {{
    font-size: {text_size}px !important;
}}

/* データフレームのテキストサイズ */
.stDataFrame {{
    font-size: {text_size}px !important;
}}

/* 通常のテキスト要素（ヘッダーを除く） */
p {{
    font-size: {text_size}px !important;
}}

/* ラベルテキスト */
label {{
    font-size: {text_size}px !important;
}}

/* 一般的なdivとspan（ただしヘッダー内のものは除外） */
div:not(.stTitle):not(.stHeader):not(.stSubheader) > span {{
    font-size: {text_size}px !important;
}}

/* 情報メッセージのテキストサイズ */
.stInfo, .stSuccess, .stWarning, .stError {{
    font-size: {text_size}px !important;
}}

/* ファイルアップローダーのテキスト */
.stFileUploader > div > div > div > div {{
    font-size: {text_size}px !important;
}}

/* スピナーのテキスト */
.stSpinner > div {{
    font-size: {text_size}px !important;
}}

/* 通常のテキスト表示 */
.stText {{
    font-size: {text_size}px !important;
}}

/* マークダウンの通常テキスト（ヘッダーは除外） */
.stMarkdown p {{
    font-size: {text_size}px !important;
}}

/* サイドバーの要素 */
.css-1d391kg p, .css-1d391kg span:not(.css-10trblm) {{
    font-size: {text_size}px !important;
}}

/* サイドバーのボタン */
.css-1d391kg .stButton > button {{
    font-size: {text_size}px !important;
}}

/* サイドバーのセレクトボックス */
.css-1d391kg .stSelectbox > div > div {{
    font-size: {text_size}px !important;
}}

/* サイドバーのスライダーラベル */
.css-1d391kg .stSlider > label {{
    font-size: {text_size}px !important;
}}

/* サイドバーのヘルプテキスト */
.css-1d391kg .stSlider > div > div > div > div {{
    font-size: {text_size}px !important;
}}

/* サイドバーの一般的なテキスト要素 */
.css-1d391kg div:not([data-testid="stSidebarNav"]) {{
    font-size: {text_size}px !important;
}}

/* サイドバーのマークダウンテキスト */
.css-1d391kg .stMarkdown p {{
    font-size: {text_size}px !important;
}}

/* サイドバーの情報メッセージ */
.css-1d391kg .stInfo, .css-1d391kg .stSuccess, .css-1d391kg .stWarning, .css-1d391kg .stError {{
    font-size: {text_size}px !important;
}}

/* より具体的なサイドバーセレクター */
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] div:not([data-testid="stSidebarNav"]):not(.stSlider):not(.stSelectbox):not(.stButton) {{
    font-size: {text_size}px !important;
}}

/* サイドバーのラベル要素 */
[data-testid="stSidebar"] label {{
    font-size: {text_size}px !important;
}}
</style>
""")

# ページ分岐
if page == "🔍 画像解析":
    # 画像解析ページ
    st.header("🔍 画像解析")
    st.write("画像をアップロードして、ML推論またはClaude解析を実行できます")
    
    # 画像アップロード機能
    uploaded_file = st.file_uploader(
        "画像を選択してください", 
        type=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    )

    if uploaded_file is not None:
        # アップロードされた画像を表示
        st.subheader("アップロードされた画像")
        
        # PILイメージとして読み込み（ML推論用）
        image = Image.open(uploaded_file)
        
        # Claude解析用：10%トリミング処理
        width, height = image.size
        crop_margin = 0.1  # 10%のマージン
        left = int(width * crop_margin)
        top = int(height * crop_margin)
        right = int(width * (1 - crop_margin))
        bottom = int(height * (1 - crop_margin))
        
        # トリミングされた画像を作成
        cropped_image = image.crop((left, top, right, bottom))
        
        # トリミングされた画像をバイナリ形式に変換
        img_buffer = io.BytesIO()
        cropped_image.save(img_buffer, format='JPEG')
        image_bytes_cropped = img_buffer.getvalue()
        
        # セッション状態に両方の形式で画像を保存
        st.session_state.current_image = image  # PIL形式（ML推論用・オリジナル）
        st.session_state.current_image_bytes = image_bytes_cropped  # バイナリ形式（Claude解析用・トリミング済み）
        
        # 画像を中央に表示
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            st.image(image, caption="アップロードされた画像", width='stretch')
        
        # 左右2分割のレイアウト（ML推論とClaude解析）
        left_col, right_col = st.columns([1, 1])
        
        # 左側：ML推論
        with left_col:
            st.subheader("🔍 ML推論")
            st.write("機械学習モデルによる高速な正常・異常判定")
            
            if st.button("推論実行", use_container_width=True):
                with st.spinner("推論中..."):
                    try:
                        # predict関数を呼び出し（クライアントを渡す）
                        result = predict(image, client)
                        
                        # セッション状態に結果を保存
                        st.session_state.current_result = result
                        
                        # 結果を表示
                        st.success("推論が完了しました！")
                        
                        # 結果の詳細を表示
                        if result:                    
                            df = pd.DataFrame(result.get("predictions"))
                            st.dataframe(df)
                            
                            # 異常が検出された場合の表示
                            if result.get("predictions") and len(result["predictions"]) > 0:
                                prediction = result["predictions"][0]
                                label = prediction.get("label", "unknown")
                                
                                if label == "damaged":
                                    st.warning("⚠️ 異常が検出されました！")
                                else:
                                    st.success("✅ 正常な製品です")
                        else:
                            st.write("推論結果が取得できませんでした")
                            
                    except Exception as e:
                        st.error(f"エラーが発生しました: {str(e)}")
                        st.write("エラーの詳細:")
                        st.exception(e)
        
        # 右側：Claude解析
        with right_col:
            st.subheader("🤖 Claude解析")
            st.write("Claude Sonnet 4による詳細な画像分析")
            
            # デフォルトの質問
            default_question = "このプリント基盤は異常品ですが、どこに異常がありますか？"
            
            # ユーザーが質問を入力できるテキストエリア
            user_question = st.text_area(
                "Claudeへの質問を入力してください:",
                value=default_question,
                height=150,
                help="画像について聞きたいことを自由に入力してください"
            )
            
            # Claude判定ボタン
            if st.button("解析実行", use_container_width=True):
                if user_question.strip():
                    with st.spinner("Claude Sonnet 4で画像を分析中..."):
                        try:
                            analysis_result = chat_with_image_and_text(
                                image=st.session_state.current_image_bytes,
                                user_question=user_question
                            )
                            
                            # 結果を表示
                            st.success("Claude Sonnet 4による分析が完了しました！")
                            
                            # 結果をより見やすく表示
                            st.markdown("### 分析結果")
                            st.write(analysis_result)
                            
                        except Exception as e:
                            st.error(f"Claude分析エラー: {str(e)}")
                            st.write("エラーの詳細:")
                            st.exception(e)
                else:
                    st.warning("質問を入力してください。")
    
    else:
        st.info("画像をアップロードしてください")

elif page == "💬 AIエージェントチャット":
    # AIエージェントチャットページ
    accept_input = False
    st.header("💬 AIエージェントチャット")
    st.write("外観検査に関する質問や、検出された異常の詳細分析を行います")
    
    # チャット履歴表示エリア（画面最大高さ）
    # 画面の高さを動的に計算してチャットエリアの高さを設定
    chat_height = 800  # デフォルトの高さ
    
    # CSSでスクロール機能を強化
    st.html("""
    <style>
    /* チャットメッセージのスタイル調整 */
    [data-testid="chatMessage"] {
        margin-bottom: 0.5rem;
    }
    
    /* チャットコンテナの枠線とパディング */
    .element-container:has([data-testid="chatMessage"]) {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        background-color: #fafafa;
        padding: 0.5rem;
    }
    </style>
    """)
    
    chat_container = st.container(height=chat_height)
    
    with chat_container:
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                if chat["type"] == "user":
                    st.chat_message("user").write(chat["content"])
                else:
                    st.chat_message("assistant").write(chat["content"])
            
            # 最新メッセージまでスクロールするための要素を追加
            st.write("")  # 空のスペースを追加してスクロール位置を調整
            
            # 自動スクロール用のJavaScript - シンプルで確実な方法
            st.html("""
            <script>
            function scrollToBottom() {
                // チャットメッセージがあるコンテナを探す
                var chatMessages = document.querySelectorAll('[data-testid="chatMessage"]');
                if (chatMessages.length > 0) {
                    // 最後のメッセージを表示エリアに持ってくる
                    var lastMessage = chatMessages[chatMessages.length - 1];
                    lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    
                    // 親コンテナも確認してスクロール
                    var parentContainer = lastMessage.closest('[data-testid="stVerticalBlock"]');
                    if (parentContainer) {
                        parentContainer.scrollTop = parentContainer.scrollHeight;
                    }
                }
            }
            
            // 少し遅延してスクロール実行
            setTimeout(scrollToBottom, 100);
            setTimeout(scrollToBottom, 500); // 追加の遅延でより確実に
            </script>
            """)
        else:
            st.info("チャットを開始してください。外観検査に関する質問や、異常原因の調査ができます。")
    
    # チャット入力エリア
    user_input = st.chat_input("メッセージを入力してください...")
    
    if user_input:
        # ユーザーメッセージを履歴に追加
        st.session_state.chat_history.append({"type": "user", "content": user_input})
        st.session_state.user_input = user_input
        st.session_state.accept_input = True  # Session Stateに保存
        st.rerun()
        
    if st.session_state.get('accept_input', False):
        # エージェントの応答を生成
        with st.spinner("AIエージェントが回答中..."):
            try:
                user_input = st.session_state.get('user_input', "")
                agent_response = investigate_anomaly_with_agent(user_question=user_input)
                
                # エージェントの応答を履歴に追加
                st.session_state.chat_history.append({"type": "assistant", "content": agent_response})
                
            except Exception as e:
                error_message = f"エラーが発生しました: {str(e)}"
                st.session_state.chat_history.append({"type": "assistant", "content": error_message})
        
        # 最後にrerunして更新されたチャット履歴を表示
        st.session_state.accept_input = False
        st.rerun()
    
    # チャット履歴クリアボタン
    if st.button("🗑️ チャット履歴をクリア"):
        st.session_state.chat_history = []
        st.rerun()

# サイドバーに接続状態を表示
st.sidebar.header("接続状態")
if client is not None:
    st.sidebar.success("✅ Databricks接続済み")
else:
    st.sidebar.warning("⚠️ デモモード")
    st.sidebar.info("環境変数を設定してください")

st.sidebar.header("使用方法")
if page == "🔍 画像解析":
    st.sidebar.write("""
    **画像解析ページ**
    1. 画像をアップロード
    2. 左側：ML推論で高速判定
    3. 右側：Claude解析で詳細分析
    4. 結果を確認
    """)
elif page == "💬 AIエージェントチャット":
    st.sidebar.write("""
    **AIエージェントチャット**
    1. 自由に質問を入力
    2. 外観検査に関する相談
    3. 専門的なアドバイス
    4. 詳細分析の依頼
    """)

# 画像解析ページでのみ対応ファイル形式を表示
if page == "🔍 画像解析":
    st.sidebar.header("対応ファイル形式")
    st.sidebar.write("JPG, PNG")
