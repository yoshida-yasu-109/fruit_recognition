# 以下を「app.py」に書き込み
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# --- 1. モデル構造の定義 (Colabの学習時と完全に一致させる) ---
def create_transfer_model(num_classes):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

@st.cache_resource
def load_my_model():
    # クラス数は 5 (Apple, Banana, Kiwi, Pineapple, Strawberry)
    model = create_transfer_model(5)
    # 重みファイルを読み込む (ファイル名はColabで保存したものに合わせる)
    model.load_weights('fruit_model_weights.weights.h5')
    return model

# モデルのロード
try:
    model = load_my_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。ファイル名を確認してください: {e}")

# --- 2. 設定データ ---
class_names = ['Apple', 'Banana', 'Kiwi', 'Pineapple', 'Strawberry']

jp_labels = {
    'Apple': 'りんご', 'Banana': 'バナナ', 'Kiwi': 'キウイ',
    'Pineapple': 'パイナップル', 'Strawberry': 'いちご'
}

fruit_descriptions = {
    "Apple": "りんごは甘みと酸味のバランスが良く、食物繊維やビタミンCが豊富な果物です。",
    "Banana": "バナナはエネルギー源として優秀で、カリウムが豊富に含まれています。",
    "Kiwi": "キウイはビタミンCが非常に豊富で、爽やかな酸味が特徴です。",
    "Pineapple": "パイナップルは甘酸っぱく、消化を助ける酵素を含んでいます。",
    "Strawberry": "いちごはビタミンCが豊富で、甘くて香り高い人気の果物です。"
}

IMG_SIZE = 224

# --- 3. UI 構築 ---
st.sidebar.title("フルーツ分類アプリ")
st.sidebar.write("画像認識モデルを使ってフルーツを分類します。")

img_source = st.sidebar.radio("画像のソースを選択してください。", ("画像をアップロード", "カメラで撮影"))

if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

# --- 4. 推論処理 ---
if img_file is not None:
    with st.spinner("推定中..."):
        # 画像の読み込みと前処理
        img = Image.open(img_file).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        st.image(img, caption="対象の画像", use_container_width=True)
        
        img_array = np.array(img_resized)
        # MobileNetV2専用の前処理を適用
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # 予測実行
        prediction = model.predict(img_array)
        probs = prediction[0]
        
        max_prob = np.max(probs)
        top_index = np.argmax(probs)
        top_fruit_en = class_names[top_index]
        
        threshold = 0.6  # しきい値

        if max_prob < threshold:
            st.error("⚠ これは登録されていない果物、あるいは判別不能な画像です。")
        else:
            st.success(f"結果: **{jp_labels[top_fruit_en]}** (確信度: {max_prob*100:.2f}%)")
            
            # 詳細（Top3）
            with st.expander("詳細な予測確率"):
                top3_indices = np.argsort(probs)[-3:][::-1]
                for i in top3_indices:
                    st.write(f"{jp_labels[class_names[i]]}: {probs[i]*100:.2f}%")
            
            # 豆知識の表示
            st.markdown("---")
            st.subheader(f"{jp_labels[top_fruit_en]}の豆知識")
            st.info(fruit_descriptions[top_fruit_en])
