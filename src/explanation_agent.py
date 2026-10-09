

import ollama


def explain_prediction(
    predicted_class,
    probability,
    feature_contributions,
):
    """Generate an explanation using a local Ollama LLM."""

    evidence = (
        ", ".join(feature_contributions)
        if feature_contributions
        else "No positive feature contributions were identified."
    )

    fallback = (
        f"The model predicted {predicted_class} with a maximum class "
        f"probability of {probability:.1%}. "
        f"Important model evidence: {evidence}. "
        "This is model-based evidence, not proof of causation."
    )

    try:
        response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain machine-learning predictions "
                        "in clear language. Use only the provided "
                        "evidence. Do not invent facts or claim "
                        "that feature contributions prove causation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Predicted class: {predicted_class}\n"
                        f"Maximum class probability: {probability:.1%}\n"
                        f"Top model contributions: {evidence}\n\n"
                        "Explain this prediction in 3-4 sentences "
                        "for a non-technical user."
                    ),
                },
            ],
            options={"temperature": 0.2},
        )

        answer = response["message"]["content"].strip()

        if answer:
            return answer, "Ollama LLM"

        return fallback, "Fallback"

    except Exception as exc:
        print(f"Ollama error: {exc}")
        return fallback, "Fallback"
