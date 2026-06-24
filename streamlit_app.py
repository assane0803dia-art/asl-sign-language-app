import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# ── Configuration ──────────────────────────────────────────────────────────────
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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS personnalisé ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    .stAlert { border-radius: 12px; }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-bottom: 1rem;
    }
    .prediction-letter {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -2px;
    }
    .prediction-label {
        font-size: 0.9rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.25);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Chargement du modèle ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

# ── En-tête ────────────────────────────────────────────────────────────────────
st.title("🤟 Reconnaissance de l'Alphabet ASL")
st.caption("Classifieur MobileNetV2 · 29 classes · American Sign Language")

if model_error:
    st.error(f"❌ Impossible de charger le modèle : `{model_error}`")
    st.stop()

st.divider()

# ── Zone de téléversement ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Téléversez une image de signe ASL",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptés : JPG, JPEG, PNG"
)

# ── Traitement ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire l'image : {e}")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Image")
        st.image(image, use_container_width=True)
        st.caption(
            f"Dimensions : {image.width} × {image.height} px  |  "
            f"Mode : {image.mode}"
        )

    # ── Prétraitement ──────────────────────────────────────────────────────────
    with st.spinner("Analyse en cours…"):
        img = image.convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100

    # ── Résultat principal ─────────────────────────────────────────────────────
    with col2:
        st.subheader("Résultat")

        # Carte de prédiction
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-letter">{predicted_class}</div>
            <div class="prediction-label">Lettre prédite</div>
            <div class="confidence-badge">{confidence:.1f}% de confiance</div>
        </div>
        """, unsafe_allow_html=True)

        # Indicateur de fiabilité
        if confidence >= 90:
            st.success("✅ Prédiction très fiable")
        elif confidence >= 70:
            st.warning("⚠️ Prédiction modérée — vérifiez l'éclairage ou le cadrage")
        else:
            st.error("❌ Confiance faible — essayez une image plus nette")

    # ── Top 5 ──────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Top 5 des prédictions")

    top5_indices = np.argsort(prediction[0])[::-1][:5]
    top5_labels = [CLASS_NAMES[i] for i in top5_indices]
    top5_values = [float(prediction[0][i]) * 100 for i in top5_indices]

    # Couleurs : première barre en violet, les autres en gris
    colors = ["#764ba2"] + ["#9ca3af"] * 4

    fig_top5 = go.Figure(go.Bar(
        x=top5_values,
        y=top5_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in top5_values],
        textposition="outside",
        hovertemplate="%{y} : %{x:.2f}%<extra></extra>",
    ))
    fig_top5.update_layout(
        margin=dict(l=10, r=60, t=10, b=10),
        height=220,
        xaxis=dict(range=[0, max(top5_values) * 1.25], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
    )
    st.plotly_chart(fig_top5, use_container_width=True, key="top5_chart")

    # ── Distribution complète ──────────────────────────────────────────────────
    with st.expander("Voir la distribution complète des probabilités"):
        all_labels = CLASS_NAMES
        all_values = [float(prediction[0][i]) * 100 for i in range(len(CLASS_NAMES))]

        bar_colors = [
            "#764ba2" if i == predicted_index else "#d1d5db"
            for i in range(len(CLASS_NAMES))
        ]

        fig_all = go.Figure(go.Bar(
            x=all_labels,
            y=all_values,
            marker_color=bar_colors,
            hovertemplate="%{x} : %{y:.2f}%<extra></extra>",
        ))
        fig_all.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            xaxis=dict(tickfont=dict(size=11)),
            yaxis=dict(title="Probabilité (%)", gridcolor="rgba(128,128,128,0.15)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_all, use_container_width=True, key="full_chart")

else:
    # ── État vide ──────────────────────────────────────────────────────────────
    st.info(
        "📁 Téléversez une image JPG ou PNG d'un signe de l'alphabet ASL "
        "pour obtenir une prédiction instantanée."
    )

    with st.expander("ℹ️ Classes reconnues par ce modèle"):
        letter_cols = st.columns(10)
        for idx, name in enumerate(CLASS_NAMES):
            letter_cols[idx % 10].markdown(
                f"<div style='text-align:center; padding:4px; "
                f"background:#f3f4f6; border-radius:6px; margin:2px; "
                f"font-weight:600;'>{name}</div>",
                unsafe_allow_html=True
            )
