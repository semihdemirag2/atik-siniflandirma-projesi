from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# Windows konsolunda Türkçe karakterlerin daha sorunsuz görünmesi için.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "TrashPanda"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "trashpanda_mobilenetv2_model.h5"

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
BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SPLIT = 0.20
RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
RESAMPLE_FILTER = getattr(Image, "Resampling", Image).LANCZOS


def configure_tensorflow_device() -> None:
    """GPU varsa TensorFlow'un otomatik kullanmasına izin verir."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("GPU bulunamadı. Eğitim CPU ile devam edecek.")
        return

    print(f"GPU bulundu: {len(gpus)} adet. TensorFlow GPU ile çalışacak.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"GPU bellek ayarı yapılamadı: {exc}")


def find_images(folder: Path) -> list[Path]:
    return [
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def collect_dataset() -> tuple[list[str], list[int], dict[str, int]]:
    """Sadece 6 ana sınıfı okur; validation klasörünü eğitim sınıfı yapmaz."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset klasörü bulunamadı: {DATASET_DIR}\n"
            "Lütfen TrashPanda klasörünü dataset klasörü altına yerleştirin."
        )

    image_paths: list[str] = []
    labels: list[int] = []
    class_counts: dict[str, int] = {}
    missing_or_empty: list[str] = []

    print("\nEğitimde kullanılacak sınıflar ve görsel sayıları")
    print("-" * 55)

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_DIR / class_name
        turkish_name = CLASS_TRANSLATIONS[class_name]

        if not class_dir.exists() or not class_dir.is_dir():
            print(f"UYARI: {class_name} ({turkish_name}) klasörü eksik.")
            class_counts[class_name] = 0
            missing_or_empty.append(class_name)
            continue

        files = sorted(find_images(class_dir))
        class_counts[class_name] = len(files)
        print(f"{class_name:12s} | {turkish_name:24s} | {len(files):6d} görsel")

        if not files:
            missing_or_empty.append(class_name)
            continue

        image_paths.extend(str(file_path) for file_path in files)
        labels.extend([label_index] * len(files))

    print("-" * 55)
    print(f"Toplam görsel sayısı: {len(image_paths)}")
    print("Not: dataset/TrashPanda/validation klasörü eğitim sınıfı olarak kullanılmadı.")

    if missing_or_empty:
        raise ValueError(
            "Eğitim başlatılamadı. Eksik veya boş sınıf klasörleri var: "
            + ", ".join(missing_or_empty)
        )

    if len(image_paths) == 0:
        raise ValueError("Desteklenen uzantıya sahip görsel bulunamadı.")

    return image_paths, labels, class_counts


def load_image_with_pillow(path_tensor: tf.Tensor) -> np.ndarray:
    """Pillow ile jpg, jpeg, png, bmp ve webp görsellerini okur."""
    image_path = path_tensor.numpy().decode("utf-8")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, RESAMPLE_FILTER)
        image_array = np.asarray(image, dtype=np.float32)

    return tf.keras.applications.mobilenet_v2.preprocess_input(image_array)


def preprocess_sample(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.py_function(load_image_with_pillow, [path], Tout=tf.float32)
    image.set_shape((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    one_hot_label = tf.one_hot(label, depth=len(CLASS_NAMES))
    return image, one_hot_label


def build_tf_dataset(paths: list[str], labels: list[int], training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        buffer_size = min(len(paths), 1000)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(preprocess_sample, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_model() -> tf.keras.Model:
    print("\nMobileNetV2 tabanlı transfer learning modeli hazırlanıyor...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_graphs(history: tf.keras.callbacks.History) -> None:
    accuracy_path = OUTPUTS_DIR / "accuracy_graph.png"
    loss_path = OUTPUTS_DIR / "loss_graph.png"

    plt.figure(figsize=(9, 6))
    plt.plot(history.history["accuracy"], marker="o", label="Training accuracy")
    plt.plot(history.history["val_accuracy"], marker="o", label="Validation accuracy")
    plt.title("Training ve Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history.history["loss"], marker="o", label="Training loss")
    plt.plot(history.history["val_loss"], marker="o", label="Validation loss")
    plt.title("Training ve Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    print(f"Accuracy grafiği kaydedildi: {accuracy_path}")
    print(f"Loss grafiği kaydedildi: {loss_path}")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    output_path = OUTPUTS_DIR / "confusion_matrix.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=35, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Tahmin edilen sınıf")
    plt.ylabel("Gerçek sınıf")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix kaydedildi: {output_path}")


def save_reports(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    classification_report_path = OUTPUTS_DIR / "classification_report.txt"
    classification_report_path.write_text(report_text, encoding="utf-8")
    print(f"Classification report kaydedildi: {classification_report_path}")

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    best_class = max(
        CLASS_NAMES,
        key=lambda class_name: report_dict[class_name]["f1-score"],
    )
    weakest_class = min(
        CLASS_NAMES,
        key=lambda class_name: report_dict[class_name]["f1-score"],
    )

    summary_text = f"""TrashPanda MobileNetV2 Eğitim Özeti
====================================

Genel accuracy: {accuracy:.4f}

Macro precision: {macro_precision:.4f}
Macro recall:    {macro_recall:.4f}
Macro F1-score:  {macro_f1:.4f}

Weighted precision: {weighted_precision:.4f}
Weighted recall:    {weighted_recall:.4f}
Weighted F1-score:  {weighted_f1:.4f}

Kısa Türkçe yorum:
MobileNetV2 tabanlı transfer öğrenme modeli, doğrulama verisi üzerinde {accuracy:.2%} genel doğruluk elde etmiştir.
Macro F1-score değeri {macro_f1:.4f} olduğundan modelin sınıflar arası ortalama başarımı bu değer üzerinden yorumlanabilir.
Weighted F1-score değeri {weighted_f1:.4f} olup sınıf destek sayılarını dikkate alan genel performansı göstermektedir.
F1-score açısından en başarılı sınıf {best_class} ({CLASS_TRANSLATIONS[best_class]}), en çok geliştirme gerektiren sınıf ise {weakest_class} ({CLASS_TRANSLATIONS[weakest_class]}) olarak gözlenmiştir.
"""

    summary_path = OUTPUTS_DIR / "metrics_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Metrik özeti kaydedildi: {summary_path}")


def main() -> None:
    print("TrashPanda atık sınıflandırma modeli eğitimi başlatılıyor.")
    print(f"Proje klasörü: {PROJECT_ROOT}")
    print(f"Dataset klasörü: {DATASET_DIR}")
    print(f"Görsel boyutu: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epoch: {EPOCHS}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    configure_tensorflow_device()
    image_paths, labels, _ = collect_dataset()

    stratify_labels = labels
    class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if np.any(class_counts < 2):
        print("UYARI: Bazı sınıflarda 2'den az görsel var. Stratified split kapatıldı.")
        stratify_labels = None

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )

    print("\nDataset train/validation olarak ayrıldı.")
    print(f"Train görsel sayısı: {len(train_paths)}")
    print(f"Validation görsel sayısı: {len(val_paths)}")

    train_dataset = build_tf_dataset(train_paths, train_labels, training=True)
    val_dataset = build_tf_dataset(val_paths, val_labels, training=False)

    model = build_model()
    model.summary()

    print("\nEğitim başlıyor. Veri seti büyükse bu işlem uzun sürebilir...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=1,
    )

    print("\nEğitim tamamlandı. Model kaydediliyor...")
    model.save(MODEL_PATH)
    print(f"Model kaydedildi: {MODEL_PATH}")

    print("\nEğitim grafikleri ve metrik dosyaları oluşturuluyor...")
    plot_training_graphs(history)

    prediction_probabilities = model.predict(val_dataset, verbose=1)
    y_pred = np.argmax(prediction_probabilities, axis=1)
    y_true = np.asarray(val_labels)

    save_confusion_matrix(y_true, y_pred)
    save_reports(y_true, y_pred)

    print("\nTüm işlemler tamamlandı.")
    print(f"Çıktılar: {OUTPUTS_DIR}")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nHATA: {exc}")
        sys.exit(1)
