from flask import Flask, request, render_template_string
import torch
import sentencepiece as spm
from model.transformer import KharadaTransformer

app = Flask(__name__)

device = torch.device("cpu")

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

vocab_size = sp.get_piece_size()
pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()

# Load model
model = KharadaTransformer(vocab_size=vocab_size).to(device)
model.load_state_dict(torch.load("model/best_model.pt", map_location=device))
model.eval()


def translate(sentence, max_len=40):

    src_ids = [bos_id] + sp.encode(sentence, out_type=int) + [eos_id]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    tgt_ids = [bos_id]

    for _ in range(max_len):

        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, pad_id)

        logits = output[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).item()

        tgt_ids.append(next_token)

        if next_token == eos_id:
            break

    # Remove BOS
    output_ids = tgt_ids[1:]

    # Remove EOS
    if eos_id in output_ids:
        output_ids = output_ids[:output_ids.index(eos_id)]

    if len(output_ids) == 0:
        return "Translation not confident."

    return sp.decode(output_ids)


HTML = """
<h2>Kharada → English Translator</h2>
<form method="post">
<input name="text" style="width:300px;">
<input type="submit">
</form>
<p><b>Translation:</b> {{ result }}</p>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    if request.method == "POST":
        text = request.form["text"]
        result = translate(text)
    return render_template_string(HTML, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)