# 🩻 Russian Neuro-Radiologist AI (Мультимодальная нейросеть)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Try%20Live%20Demo-blue)](https://huggingface.co/spaces/livadies/AI-Radiologist-RU)
*(Кликните по бейджу выше, чтобы протестировать нейросеть прямо в браузере)*

**End-to-End проект** по созданию русскоязычной мультимодальной нейросети (Vision-Language Model), способной генерировать медицинские заключения на естественном русском языке на основе рентгеновских снимков грудной клетки (Chest X-Ray).

---

## 🌟 О проекте и проблематике
Большинство открытых медицинских датасетов (MIMIC-CXR, CheXpert) и генеративных моделей работают исключительно с английским языком. Цель данного проекта — создать **первую открытую русскоязычную архитектуру "Радиолог-Ассистент"**, объединив компьютерное зрение (CV) и обработку естественного языка (NLP).

### Что умеет модель:
- 👁️ **Визуальный анализ:** Извлечение паттернов заболеваний (пневмоторакс, консолидация, кардиомегалия) с рентгеновского снимка.
- ✍️ **Генерация текста:** Написание связного, структурированного радиологического отчета на профессиональном медицинском русском языке.

---

## 🧠 Архитектура (VisionEncoderDecoder)
Модель построена путем объединения двух state-of-the-art архитектур через слои Cross-Attention:
*   **Vision Encoder (Глаза):** `google/vit-base-patch16-224-in21k` (Vision Transformer). Переводит пиксели в семантические векторы.
*   **Text Decoder (Мозг):** `ai-forever/rugpt3small_based_on_gpt2` (ruGPT-3 от Сбера). Конфигурация модели была модифицирована (`is_decoder=True`, `add_cross_attention=True`) для генерации русского текста на основе визуальных эмбеддингов от ViT.

---

## 🛠 Data Engineering Pipeline
Так как готовых русскоязычных пар "X-Ray -> Русский текст" в открытом доступе нет, был реализован хардкорный пайплайн авто-перевода и создания датасета:
1. **Сбор данных:** Использование открытого датасета **Indiana University Chest X-Ray (IU X-Ray)** (более 7000 снимков).
2. **Translation Pipeline:** Развертывание модели машинного перевода `Helsinki-NLP/opus-mt-en-ru` на GPU с использованием Batching для массового перевода тысяч английских медицинских заключений.
3. **Custom PyTorch Dataset:** Написание кастомного загрузчика с глубоким сканированием файловой системы (Deep Scan Mapping) для точной привязки файлов, ресайза (224x224) и токенизации.

---

## ⚙️ Технические детали обучения (MLOps)
Модель обучалась на кластере **2x NVIDIA T4 GPU**.

*   **Оптимизация производительности:** 
    * Mixed Precision Training (`fp16`) для ускорения обучения в 2 раза.
    * Gradient Accumulation (виртуальный батч = 16) для обхода ограничений VRAM.
*   **Стабилизация градиентов:** L2 Regularization (Weight Decay) и Cosine Warmup.
*   **Inference Strategy:** Применение Beam Search (`num_beams=4`) с `repetition_penalty=2.0` для генерации связных диагнозов без зацикливаний.

---

## 🚀 Как запустить локально
В репозитории находится Jupyter Notebook (`.ipynb`). Вы можете загрузить его в Google Colab или Kaggle, подключить GPU и воспроизвести весь пайплайн от Data Engineering до обучения мультимодальной сети.

```python
# Пример использования готовой модели (Inference)
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

model_id = "livadies/Russian-Radiologist-ruGPT-ViT"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
feature_extractor = ViTImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

image = Image.open("your_xray.png").convert("RGB")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values, max_length=128, num_beams=4)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
