import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "asl_mobilenetv2.h5"

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🤟 Reconnaissance de l'Alphabet ASL")
st.write(
    "Téléversez une image d'une lettre ASL pour obtenir une prédiction."
)

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Image téléversée",
            use_container_width=True
        )

    # Prétraitement
    img = image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prédiction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        prediction[0][predicted_index]
    ) * 100

    with col2:

        st.success(
            f"Prédiction : {predicted_class}"
        )

        st.metric(
            "Confiance",
            f"{confidence:.2f}%"
        )

        if confidence > 90:
            st.success("Prédiction très fiable")

        elif confidence > 70:
            st.warning("Prédiction correcte")

        else:
            st.error("Confiance faible")

    st.subheader("Top 5 prédictions")

    top5 = np.argsort(prediction[0])[::-1][:5]

    for idx in top5:
        st.write(
            f"{CLASS_NAMES[idx]} : "
            f"{prediction[0][idx] * 100:.2f}%"
        )

    st.subheader("Probabilités")

    probs = {
        CLASS_NAMES[i]: float(prediction[0][i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(probs)

else:
    st.info(
        "Veuillez téléverser une image."
    )


