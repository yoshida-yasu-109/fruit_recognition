# 以下を「app.py」に書き込み
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('fruit_transfer_model.keras')

class_names = ['Apple', 'Banana', 'Kiwi', 'Pineapple', 'Strawberry']

jp_labels = {
    'Apple': 'りんご',
    'Banana': 'バナナ',
    'Kiwi': 'キウイ',
    'Pineapple': 'パイナップル',
    'Strawberry': 'いちご'
}

fruit_descriptions = {
    "Apple": "りんごは甘みと酸味のバランスが良く、食物繊維やビタミンCが豊富な果物です。",
    "Banana": "バナナはエネルギー源として優秀で、カリウムが豊富に含まれています。",
    "Kiwi": "キウイはビタミンCが非常に豊富で、爽やかな酸味が特徴です。",
    "Pineapple": "パイナップルは甘酸っぱく、消化を助ける酵素を含んでいます。",
    "Strawberry": "いちごはビタミンCが豊富で、甘くて香り高い人気の果物です。"
}

IMG_SIZE = 224

st.sidebar.title("フルーツ分類アプリ")
st.sidebar.write("画像認識モデルを使って写真に写っているフルーツを分類します。")

st.sidebar.write("")

img_source = st.sidebar.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "画像をアップロード":
    img_file = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg"])
elif img_source == "カメラで撮影":
    img_file = st.camera_input("カメラで撮影")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        st.image(img, caption="対象の画像", width="stretch")
        
        img_array = np.array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        probs = prediction[0]

        top3_indices = np.argsort(probs)[-3:][::-1]
        max_prob = np.max(probs)

        threshold = 0.6

        if max_prob < threshold:
            st.error("⚠ これは登録されていない果物です（その他）")
        
        else :

            st.subheader("予測結果")

            for i in top3_indices:
                fruit_en = class_names[i]
                fruit_jp = jp_labels[fruit_en]
                confidence = probs[i] * 100
                st.write(f"{confidence:.2f}% の確率で {fruit_jp} です")
                
        
        top_index = np.argmax(probs)
        top_fruit = class_names[top_index]
        st.markdown("---")
        st.subheader("果物の豆知識")
        st.write(fruit_descriptions[top_fruit])
