# Transfer Öğrenme Destekli Atık Sınıflandırma Projesi

Bu proje, **Transfer Öğrenme Destekli Hafif CNN Mimarisi ile Atık Türlerinin Görüntü Tabanlı Sınıflandırılması** konusu için hazırlanmıştır. TrashPanda veri kümesi kullanılarak MobileNetV2 tabanlı bir derin öğrenme modeli eğitilir ve atık görselleri 6 ana sınıfa ayrılır.

## Projenin amacı

Amaç, atık türlerini görüntü üzerinden otomatik sınıflandırabilen hafif ve uygulanabilir bir CNN modeli geliştirmektir. Modelde ImageNet ağırlıklarıyla önceden eğitilmiş MobileNetV2 mimarisi kullanılır. MobileNetV2'nin temel katmanları ilk aşamada dondurulur ve sonuna sınıflandırma için GlobalAveragePooling2D, Dropout ve Dense softmax katmanları eklenir.

## Dataset yapısı

Beklenen klasör yapısı:

```text
atik_siniflandirma_projesi/
├── dataset/
│   └── TrashPanda/
│       ├── Biomuell
│       ├── GelberSack
│       ├── Glas
│       ├── Papier
│       ├── Restmuell
│       ├── Sondermuell
│       └── validation
├── code/
├── outputs/
├── models/
└── report/
```

Eğitim sırasında yalnızca 6 ana sınıf kullanılır. `validation` klasörü eğitim sınıfı olarak algılanmaz. Eğitim ve doğrulama ayrımı `train_model.py` içinde otomatik olarak yapılır.

Desteklenen görsel uzantıları: `jpg`, `jpeg`, `png`, `bmp`, `webp`.

## Sınıflar

| Klasör adı | Türkçe anlamı |
|---|---|
| Biomuell | Organik atık |
| GelberSack | Ambalaj / plastik atığı |
| Glas | Cam |
| Papier | Kağıt |
| Restmuell | Genel evsel atık |
| Sondermuell | Özel / tehlikeli atık |

## Kurulum

Önce proje ana klasörüne gidin:

```powershell
cd "C:\Users\pc\OneDrive\Masaüstü\atik_siniflandirma_projesi"
```

İsteğe bağlı olarak sanal ortam oluşturabilirsiniz:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Gerekli kütüphaneleri yükleyin:

```powershell
pip install -r requirements.txt
```

## Dataset kontrol komutu

Dataset klasörlerini, sınıf sayılarını ve `validation` klasörünün durumunu kontrol etmek için:

```powershell
python code\dataset_kontrol.py
```

Bu komut eksik sınıf, boş sınıf veya eksik dataset klasörü varsa ekrana uyarı verir.

## Model eğitim komutu

Modeli eğitmek için:

```powershell
python code\train_model.py
```

Varsayılan ayarlar:

- Görsel boyutu: `224x224`
- Batch size: `16`
- Epoch sayısı: `10`
- Model: `MobileNetV2`
- Ağırlıklar: `ImageNet`
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Metrik: `accuracy`

Eğitim sonunda model şu dosyaya kaydedilir:

```text
models/trashpanda_mobilenetv2_model.h5
```

TensorFlow GPU varsa otomatik olarak GPU kullanır. GPU yoksa CPU ile çalışır.

## Tahmin komutu

Eğitilmiş model ile tek bir görsel için tahmin yapmak için:

```powershell
python code\predict_image.py
```

Program sizden tahmin edilecek görselin tam yolunu ister. Ardından en olası sınıfı, Türkçe karşılığını ve tüm sınıfların olasılıklarını ekrana yazdırır.

## Outputs klasöründe oluşacak dosyalar

Eğitimden sonra `outputs` klasöründe şu dosyalar oluşur:

| Dosya | Açıklama |
|---|---|
| accuracy_graph.png | Training accuracy ve validation accuracy grafiği |
| loss_graph.png | Training loss ve validation loss grafiği |
| confusion_matrix.png | Sınıf isimleri eksende görünen confusion matrix |
| classification_report.txt | Precision, recall, F1-score ve support değerleri |
| metrics_summary.txt | Genel accuracy, macro ve weighted metrikler, kısa Türkçe yorum |

## Rapor için kullanılabilecek çıktılar

Ödev raporunda özellikle şu dosyalar kullanılabilir:

- `outputs/accuracy_graph.png`
- `outputs/loss_graph.png`
- `outputs/confusion_matrix.png`
- `outputs/classification_report.txt`
- `outputs/metrics_summary.txt`

Rapor metninde modelin genel başarımı için `metrics_summary.txt`, sınıf bazlı performans için `classification_report.txt`, eğitim sürecini göstermek için accuracy/loss grafikleri ve hata dağılımını göstermek için confusion matrix görseli kullanılabilir.
