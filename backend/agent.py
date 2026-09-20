import os
import json

from dotenv import load_dotenv
from google import genai


# ============================================================
# Configuration
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured in backend/.env"
    )


client = genai.Client(api_key=api_key)


# ============================================================
# System instructions
# ============================================================

SYSTEM_PROMPT = """
You are TriageAI, a healthcare triage decision-support assistant.

Your job is to explain the output of an existing machine-learning
triage classifier to a clinician or healthcare professional.

IMPORTANT RULES:

1. The XGBoost model is the actual triage classifier.
2. Never change, override, or replace the model's predicted triage class.
3. Never invent patient information.
4. Base your explanation only on the supplied patient features
   and model output.
5. Do not claim to diagnose the patient.
6. Do not prescribe medication or treatment.
7. Clearly distinguish model output from clinical judgment.
8. Highlight important submitted findings and model probabilities.
9. Mention uncertainty when probabilities are close.
10. Use concise, professional clinical language.
11. This is decision support, not a substitute for professional
    clinical assessment.

Return a structured explanation containing:

- Summary
- Why the model produced the predicted class
- Key observations from the submitted data
- Probability interpretation
- Data quality / missing information considerations
- Clinical decision-support notice

Do not invent values that are not present in the input.
"""


# ============================================================
# AI Agent
# ============================================================

def analyze_triage(
    features: dict,
    prediction: str,
    probabilities: dict,
) -> str:

    payload = {
        "model_prediction": prediction,
        "model_probabilities": probabilities,
        "patient_features": features,
    }

    prompt = f"""
{SYSTEM_PROMPT}

Here is the model output and submitted patient data:

{json.dumps(payload, indent=2)}

Generate the clinician-facing explanation now.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text