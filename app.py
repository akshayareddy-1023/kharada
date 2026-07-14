from flask import Flask, request, render_template, redirect, url_for, session, flash
import torch
import sentencepiece as spm
from model.transformer import KharadaTransformer

app = Flask(__name__)

app.secret_key = "kharada_secret_key"
USERNAME = "admin"
PASSWORD = "admin123"
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
users = []

history_data = []


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

@app.route("/")
def index():

    return render_template("login.html")



@app.route("/register", methods=["GET","POST"])
def register():

    if request.method=="POST":

        username=request.form["username"]
        email=request.form["email"]
        password=request.form["password"]


        users.append({
            "username":username,
            "email":email,
            "password":password
        })


        flash("Registration successful")

        return redirect(url_for("index"))


    return render_template("register.html")




@app.route("/login", methods=["GET","POST"])
def login():

    if request.method=="POST":

        email=request.form["email"]
        password=request.form["password"]


        for user in users:

            if user["email"]==email and user["password"]==password:

                session["user"]=user["username"]

                return redirect(url_for("home"))


        flash("Invalid login")


    return render_template("login.html")




@app.route("/translator", methods=["GET","POST"])
def home():

    result=""

    if request.method=="POST":

        text=request.form["inputText"]

        result=translate(text)


        history_data.append({

            "input_text":text,

            "output_text":result,

            "date":"Recent"

        })


    return render_template(
        "index.html",
        translation=result
    )





@app.route("/history")
def history():

    return render_template(
        "history.html",
        history=history_data
    )





@app.route("/about")
def about():

    return render_template("about.html")





@app.route("/logout")
def logout():

    session.pop("user",None)

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)