import subprocess
import sys
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ========================
# Установка зависимостей
# ========================
try:
    import transformers
except ImportError:
    print("Installing transformers...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])

try:
    import gradio
except ImportError:
    print("Installing gradio...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])

# ========================
# ЗАГРУЗКА МОДЕЛИ
# ========================
MODEL_PATH = "./model"

# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Определяем категории
categories = [
    'бытовая техника',
    'обувь',
    'одежда',
    'посуда',
    'текстиль',
    'товары для детей',
    'украшения и аксессуары',
    'электроника',
    'нет товара'
]

# ========================
# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# ========================
def predict_review(text):
    if not text.strip():
        return "Введите текст отзыва", ""

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()

    predicted_category = categories[pred_class]
    confidence_percent = round(confidence * 100, 2)

    result_text = f"🎯 Категория: **{predicted_category}**\n" \
                  f"📊 Уверенность: **{confidence_percent}%**"

    return result_text, predicted_category

# ========================
# GRADIO ИНТЕРФЕЙС
# ========================
with gr.Blocks(
    title="Классификатор отзывов — T-Банк",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    css="""
    .gradio-container { background: linear-gradient(135deg, #f5f7fa, #e4e8eb); }
    h1 { text-align: center; color: #2c3e50; font-weight: 700; }
    .footer { text-align: center; margin-top: 20px; color: #7f8c8d; font-size: 0.9em; }
    """
) as demo:
    gr.Markdown("# 🧠 Классификатор отзывов от T-Банк")
    gr.Markdown("### Определяет категорию товара по тексту отзыва")

    with gr.Row():
        with gr.Column(scale=2):
            review_input = gr.Textbox(
                label="📝 Введите отзыв",
                placeholder="Например: 'Отличные наушники, звук чистый'",
                lines=5
            )
            examples = gr.Examples(
                examples=[
                    ["Замечательный пылесос, тихий и мощный!"],
                    ["Кроссовки жмут, но выглядят стильно"],
                    ["Платье село идеально, ткань приятная"],
                    ["Сковорода пригорает, не рекомендую"],
                    ["Заказ не пришел, деньги не вернули"]
                ],
                inputs=review_input,
                label="Примеры отзывов"
            )
            classify_btn = gr.Button("🚀 Классифицировать", variant="primary")

        with gr.Column(scale=1):
            result_output = gr.Textbox(label="🎯 Результат", interactive=False)
            category_output = gr.Label(label="📈 Вероятности по категориям (топ-3)")

    classify_btn.click(
        fn=predict_review,
        inputs=review_input,
        outputs=[result_output, category_output]
    )

    gr.Markdown(
        """
        <div class="footer">
        Проект Sirius NLP Case — Matvey Tizhin | T-Банк 2025
        </div>
        """,
        elem_classes=["footer"]
    )

if __name__ == "__main__":
    demo.launch()
