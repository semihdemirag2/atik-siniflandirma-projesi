from pathlib import Path
import sys


# Windows konsolunda Türkçe karakterlerin daha sorunsuz görünmesi için.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "TrashPanda"

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


def find_images(folder: Path) -> list[Path]:
    """Desteklenen uzantılara sahip görselleri alt klasörlerle birlikte bulur."""
    return [
        file_path
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def print_validation_info(validation_dir: Path) -> None:
    print("\nValidation klasörü kontrolü")
    print("-" * 30)

    if not validation_dir.exists():
        print(f"UYARI: validation klasörü bulunamadı: {validation_dir}")
        return

    if not validation_dir.is_dir():
        print(f"UYARI: validation yolu klasör değil: {validation_dir}")
        return

    entries = sorted(validation_dir.iterdir(), key=lambda item: item.name.lower())
    if not entries:
        print("UYARI: validation klasörü boş.")
        return

    total_images = len(find_images(validation_dir))
    print(f"Toplam desteklenen görsel sayısı: {total_images}")

    for entry in entries:
        if entry.is_dir():
            image_count = len(find_images(entry))
            status = "BOŞ" if image_count == 0 else "OK"
            print(f"- {entry.name}: {image_count} görsel ({status})")
        elif entry.is_file():
            supported = entry.suffix.lower() in SUPPORTED_EXTENSIONS
            status = "destekleniyor" if supported else "desteklenmeyen uzantı"
            print(f"- {entry.name}: dosya ({status})")


def main() -> None:
    print("TrashPanda dataset kontrolü başlatıldı.")
    print(f"Proje klasörü: {PROJECT_ROOT}")
    print(f"Dataset klasörü: {DATASET_DIR}")
    print(f"Desteklenen uzantılar: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if not DATASET_DIR.exists():
        print(f"\nHATA: Dataset klasörü bulunamadı: {DATASET_DIR}")
        print("Lütfen TrashPanda klasörünü dataset klasörü altına yerleştirin.")
        return

    if not DATASET_DIR.is_dir():
        print(f"\nHATA: Dataset yolu klasör değil: {DATASET_DIR}")
        return

    print("\nAna sınıf klasörleri")
    print("-" * 30)

    total_images = 0
    missing_classes: list[str] = []
    empty_classes: list[str] = []

    for class_name in CLASS_NAMES:
        class_dir = DATASET_DIR / class_name
        turkish_name = CLASS_TRANSLATIONS[class_name]

        if not class_dir.exists():
            print(f"UYARI: {class_name} ({turkish_name}) klasörü eksik.")
            missing_classes.append(class_name)
            continue

        if not class_dir.is_dir():
            print(f"UYARI: {class_name} yolu klasör değil.")
            missing_classes.append(class_name)
            continue

        image_count = len(find_images(class_dir))
        total_images += image_count

        if image_count == 0:
            empty_classes.append(class_name)
            print(f"UYARI: {class_name} ({turkish_name}): 0 görsel")
        else:
            print(f"{class_name:12s} | {turkish_name:24s} | {image_count:6d} görsel")

    print("-" * 30)
    print(f"Ana sınıflardaki toplam görsel sayısı: {total_images}")

    validation_dir = DATASET_DIR / "validation"
    print_validation_info(validation_dir)

    extra_dirs = sorted(
        [
            item.name
            for item in DATASET_DIR.iterdir()
            if item.is_dir() and item.name not in CLASS_NAMES and item.name != "validation"
        ]
    )
    if extra_dirs:
        print("\nUYARI: Beklenen sınıflar dışında klasör bulundu:")
        for dir_name in extra_dirs:
            print(f"- {dir_name}")

    print("\nÖzet")
    print("-" * 30)
    if missing_classes:
        print(f"Eksik sınıflar: {', '.join(missing_classes)}")
    else:
        print("Eksik sınıf yok.")

    if empty_classes:
        print(f"Boş sınıflar: {', '.join(empty_classes)}")
    else:
        print("Boş ana sınıf klasörü yok.")

    print("Kontrol tamamlandı.")


if __name__ == "__main__":
    main()
