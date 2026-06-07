from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "trashpanda_mobilenetv2_model.h5"

CLASS_NAMES = [
    "Biomuell",
    "GelberSack",
    "Glas",
    "Papier",
    "Restmuell",
    "Sondermuell",
]

CLASS_TRANSLATIONS = {
    "Biomuell": "Organik Atık",
    "GelberSack": "Ambalaj / Plastik Atığı",
    "Glas": "Cam",
    "Papier": "Kağıt",
    "Restmuell": "Genel Evsel Atık",
    "Sondermuell": "Özel / Tehlikeli Atık",
}

DISPLAY_NAMES = {
    class_name: f"{turkish_name} ({class_name})"
    for class_name, turkish_name in CLASS_TRANSLATIONS.items()
}

IMAGE_SIZE = (224, 224)
IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]
HTML_FORMATS = ["html", "htm"]
SUPPORTED_FORMATS = IMAGE_FORMATS + HTML_FORMATS
RESAMPLE_FILTER = getattr(Image, "Resampling", Image).LANCZOS


st.set_page_config(
    page_title="Atık Türü Sınıflandırma Sistemi",
    page_icon="♻️",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        background: #f9fafb;
    }
    .result-label {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 0.1rem;
    }
    .result-value {
        color: #111827;
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }
    .result-line {
        color: #111827;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    .result-key {
        color: #374151;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Eğitilmiş modeli bir kez yükler ve Streamlit oturumu boyunca cache'ler."""
    import tensorflow as tf

    source_path = Path(model_path)

    # h5py bazı Windows kurulumlarında Türkçe karakter içeren yollardan .h5 okuyamayabilir.
    # Bu nedenle modeli geçici, kısa ve ASCII karakterli bir yola kopyalayarak yüklüyoruz.
    temp_dir = Path(tempfile.gettempdir()) / "trashpanda_streamlit_model"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_model_path = temp_dir / source_path.name

    if (
        not safe_model_path.exists()
        or safe_model_path.stat().st_size != source_path.stat().st_size
        or safe_model_path.stat().st_mtime < source_path.stat().st_mtime
    ):
        shutil.copy2(source_path, safe_model_path)

    return tf.keras.models.load_model(str(safe_model_path), compile=False)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Yüklenen görseli MobileNetV2 modelinin beklediği formata dönüştürür."""
    import tensorflow as tf

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, RESAMPLE_FILTER)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


def create_probability_chart(probabilities: np.ndarray) -> plt.Figure:
    """Tüm sınıf olasılıklarını okunabilir bir yatay bar chart olarak çizer."""
    percentages = probabilities * 100
    colors = ["#2563eb" if value == percentages.max() else "#94a3b8" for value in percentages]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y_positions = np.arange(len(CLASS_NAMES))

    ax.barh(y_positions, percentages, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DISPLAY_NAMES[class_name] for class_name in CLASS_NAMES])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Olasılık (%)")
    ax.set_title("Sınıf Olasılıkları")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)

    for index, value in enumerate(percentages):
        ax.text(min(value + 1, 98), index, f"%{value:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def render_sidebar() -> None:
    st.sidebar.header("Proje Bilgileri")
    st.sidebar.metric("Model", "MobileNetV2")
    st.sidebar.metric("Dataset", "TrashPanda")
    st.sidebar.metric("Sınıf sayısı", "6")
    st.sidebar.metric("Genel accuracy", "%91.59")
    st.sidebar.metric("Weighted F1-score", "%91.52")

    st.sidebar.divider()
    st.sidebar.caption("Sınıflar")
    for class_name in CLASS_NAMES:
        st.sidebar.write(f"**{CLASS_TRANSLATIONS[class_name]}** → {class_name}")


def main() -> None:
    render_sidebar()

    st.markdown('<div class="main-title">Atık Türü Sınıflandırma Sistemi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Bu uygulama, MobileNetV2 tabanlı transfer öğrenme modeli kullanarak '
        "yüklenen atık görselinin sınıfını tahmin eder.</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "Bu sistem, yüklenen atık görselini 6 farklı atık sınıfından birine ayırmak için "
        "MobileNetV2 tabanlı transfer öğrenme modeli kullanır."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Model dosyası bulunamadı. Lütfen önce modeli eğitin veya "
            f"`{MODEL_PATH}` konumuna model dosyasını yerleştirin."
        )

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Görsel Yükle")
        uploaded_file = st.file_uploader(
            "Atık görselini veya HTML dosyasını seçin",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=False,
        )

        image = None
        if uploaded_file is not None:
            uploaded_extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
            try:
                if uploaded_extension in HTML_FORMATS:
                    html_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
                    st.caption("Yüklenen HTML önizlemesi")
                    components.html(html_content, height=420, scrolling=True)
                    st.warning(
                        "HTML dosyası kabul edildi ve önizlendi. Model tahmini yapmak için "
                        "HTML içindeki görseli ayrıca jpg, jpeg, png veya webp olarak yükleyin."
                    )
                else:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Yüklenen görsel", use_container_width=True)
            except Exception as exc:
                st.error(f"Dosya okunamadı: {exc}")

        predict_clicked = st.button(
            "Tahmin Et",
            type="primary",
            use_container_width=True,
            disabled=image is None or not MODEL_PATH.exists(),
        )

        if uploaded_file is None:
            st.info("jpg, jpeg, png, webp veya html formatında bir dosya yükleyebilirsiniz.")

    with right_col:
        st.subheader("Tahmin Sonucu")

        if predict_clicked:
            if image is None:
                st.warning("Tahmin yapılacak geçerli bir görsel bulunamadı.")
                return

            try:
                with st.spinner("Model yükleniyor ve tahmin yapılıyor..."):
                    model = load_model(str(MODEL_PATH))
                    prepared_image = prepare_image(image)
                    probabilities = model.predict(prepared_image, verbose=0)[0]

                best_index = int(np.argmax(probabilities))
                best_class = CLASS_NAMES[best_index]
                best_probability = float(probabilities[best_index])

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="result-line">
                            <span class="result-key">Tahmin Edilen Sınıf:</span>
                            {CLASS_TRANSLATIONS[best_class]}
                        </div>
                        <div class="result-line">
                            <span class="result-key">Orijinal Etiket:</span>
                            {best_class}
                        </div>
                        <div class="result-line">
                            <span class="result-key">Güven Oranı:</span>
                            %{best_probability * 100:.2f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.pyplot(create_probability_chart(probabilities), use_container_width=True)

            except Exception as exc:
                st.error(f"Tahmin sırasında hata oluştu: {exc}")
        else:
            st.info("Görsel yüklendikten sonra sonucu görmek için **Tahmin Et** butonuna basın.")

    st.divider()
    st.caption(
        "Not: Model tahmini, eğitim veri setindeki görsel örüntülere göre yapılmaktadır. "
        "Gerçek atık ayrıştırma kararlarında yerel belediye kuralları dikkate alınmalıdır."
    )


if __name__ == "__main__":
    main()
