from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import date
from pathlib import Path

# --- setup ---
docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)
pdf_path = docs_dir / "model_card.pdf"

styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
story = []

def add_heading(text, level=1):
    style = styles["Heading1"] if level == 1 else styles["Heading2"]
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 8))

def add_paragraph(text):
    story.append(Paragraph(text, styles["BodyText"]))
    story.append(Spacer(1, 8))

# --- content ---
add_heading("Model Card — NL Rent Prediction (v1)")


add_heading("1. Model Overview", 2)
add_paragraph(
    "This model predicts monthly rent prices (EUR) for residential listings in the Netherlands. "
    "It was trained on a 2019–2020 Kaggle dataset of NL rental data, with an uplift factor of 1.50 applied "
    "to account for price growth. The model is deployed as a FastAPI web service with monitoring and feedback."
)

add_heading("2. Intended Use", 2)
add_paragraph(
    "This model is intended for educational and demonstration purposes only. "
    "It showcases end-to-end deployment, observability, and CI integration for a regression task."
)

add_heading("3. Model Details", 2)
add_paragraph(
    "• Algorithm: XGBoost Regressor<br/>"
    "• Features: areaSqm, latitude, longitude, propertyType, furnish, internet, kitchen, shower, "
    "toilet, living, smokingInside, pets, city_prior, pc4_prior<br/>"
    "• Frameworks: scikit-learn (pipeline), FastAPI, XGBoost<br/>"
    "• Data split: 85% train / 15% test<br/>"
)

add_heading("4. Performance", 2)
add_paragraph("Test R² ≈ 0.845, MAE ≈ €142 on the held-out validation set.")

add_heading("5. Limitations", 2)
add_paragraph(
    "• Data covers only 2019–2020 period and may not reflect current rental market.<br/>"
    "• Geographic coverage limited to the Netherlands.<br/>"
    "• Uplift factor introduces estimation bias for newer listings.<br/>"
    "• No external economic or seasonal factors considered."
)

add_heading("6. Ethical Considerations", 2)
add_paragraph(
    "Model should not be used for real rental pricing decisions or credit-related purposes. "
    "Predictions are for educational visualization only."
)

add_heading("7. Author", 2)
add_paragraph(
    "Developed by Parissa, Data Scientist.\n"
)

# --- build PDF ---
doc.build(story)
print(f"✅ Model card saved to: {pdf_path.resolve()}")
