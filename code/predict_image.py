from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


# Windows konsolunda Türkçe karakterlerin daha sorunsuz görünmesi için.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


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
    "Biomuell": "Organik atık",
    "GelberSack": "Ambalaj / plastik atığı",
    "Glas": "Cam",
    "Papier": "Kağıt",
    "Restmuell": "Genel evsel atık",
    "Sondermuell": "Özel / tehlikeli atık",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (224, 224)
RESAMPLE_FILTER = getattr(Image, "Resampling", Image).LANCZOS


def normalize_user_path(raw_path: str) -> Path:
    """Kullanıcının tırnak içinde verdiği yolu temizler."""
    cleaned_path = raw_path.strip().strip('"').strip("'")
    return Path(cleaned_path).expanduser()


def load_and_prepare_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, RESAMPLE_FILTER)
        image_array = np.asarray(image, dtype=np.float32)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


def main() -> None:
    print("TrashPanda atık türü tahmin aracı")
    print(f"Model yolu: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print("\nHATA: Eğitilmiş model dosyası bulunamadı.")
        print("Önce şu komutla modeli eğitin:")
        print("python code/train_model.py")
        return

    raw_image_path = input("\nTahmin edilecek görselin tam yolunu girin: ")
    image_path = normalize_user_path(raw_image_path)

    if not image_path.exists():
        print(f"\nHATA: Görsel dosyası bulunamadı: {image_path}")
        return

    if not image_path.is_file():
        print(f"\nHATA: Verilen yol dosya değil: {image_path}")
        return

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            "\nHATA: Desteklenmeyen görsel uzantısı. "
            f"Desteklenen uzantılar: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    print("\nModel yükleniyor...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Görsel hazırlanıyor ve tahmin yapılıyor...")
    prepared_image = load_and_prepare_image(image_path)
    probabilities = model.predict(prepared_image, verbose=0)[0]

    best_index = int(np.argmax(probabilities))
    best_class = CLASS_NAMES[best_index]
    best_probability = probabilities[best_index]

    print("\nTahmin sonucu")
    print("-" * 30)
    print(f"En olası sınıf: {best_class} ({CLASS_TRANSLATIONS[best_class]})")
    print(f"Olasılık: {best_probability:.4f} ({best_probability * 100:.2f}%)")

    print("\nTüm sınıf olasılıkları")
    print("-" * 30)
    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(
            f"{class_name:12s} | {CLASS_TRANSLATIONS[class_name]:24s} | "
            f"{probability:.4f} ({probability * 100:.2f}%)"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nHATA: {exc}")
        sys.exit(1)
