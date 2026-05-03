from flask import Flask, render_template, request

from spell_core import correct_sentence, load_spell_model

app = Flask(__name__)
model, char2idx, idx2char, lookup_tables = load_spell_model()

@app.route("/", methods=["GET", "POST"])
def index():
    original = ""
    corrected = ""
    mistakes = []
    status = ""

    if request.method == "POST":
        original = request.form.get("sentence", "")
        mistakes, corrected = correct_sentence(original, model, char2idx, idx2char, lookup_tables)
        if original.strip() and not mistakes:
            status = "Sentence correct"

    return render_template(
        "index.html",
        original=original,
        corrected=corrected,
        mistakes=mistakes,
        status=status,
    )

if __name__ == "__main__":
    app.run(debug=True)
