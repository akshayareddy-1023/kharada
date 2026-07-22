from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify
)

import sqlite3
import json
import torch
import sentencepiece as spm
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from model.transformer import KharadaTransformer
from model.lstm import KharadaLSTM


app = Flask(__name__)
app.secret_key = "kharada_secret_key"

DATABASE = "kharada.db"
import os

print("Database exists:", os.path.exists(DATABASE))
print("Database path:", os.path.abspath(DATABASE))
DEVICE = torch.device("cpu")
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn
def create_database():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        email TEXT UNIQUE,
        password TEXT,
        login_time TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        model TEXT,
        input_text TEXT,
        translation TEXT,
        translated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
create_database()

metrics = {

    "Transformer":{

        "BLEU":4.04,
        "Accuracy":82,
        "Precision":81,
        "Recall":80,
        "F1":80

    },

    "LSTM":{

        "BLEU":2.80,
        "Accuracy":75,
        "Precision":74,
        "Recall":73,
        "F1":73

    }

}
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

VOCAB_SIZE = sp.get_piece_size()

PAD_ID = sp.pad_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
transformer_model = KharadaTransformer(
    vocab_size=VOCAB_SIZE
).to(DEVICE)

transformer_model.load_state_dict(
    torch.load(
        "model/best_model.pt",
        map_location=DEVICE
    )
)

transformer_model.eval()
lstm_model = KharadaLSTM(
    vocab_size=VOCAB_SIZE
).to(DEVICE)

lstm_model.load_state_dict(
    torch.load(
        "model/best_lstm.pt",
        map_location=DEVICE
    )
)

lstm_model.eval()
# -------------------------------------------------------
# Transformer Translation
# -------------------------------------------------------

def transformer_translate(sentence, max_len=40):

    src_ids = [BOS_ID] + sp.encode(sentence, out_type=int) + [EOS_ID]

    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

    tgt_ids = [BOS_ID]

    for _ in range(max_len):

        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = transformer_model(src_tensor, tgt_tensor, PAD_ID)

        next_token = output[:, -1, :].argmax(dim=-1).item()

        tgt_ids.append(next_token)

        if next_token == EOS_ID:
            break

    output_ids = tgt_ids[1:]

    if EOS_ID in output_ids:
        output_ids = output_ids[:output_ids.index(EOS_ID)]

    if len(output_ids) == 0:
        return "Translation not found"

    return sp.decode(output_ids)


# -------------------------------------------------------
# LSTM Translation
# -------------------------------------------------------

def lstm_translate(sentence, max_len=40):

    src_ids = [BOS_ID] + sp.encode(sentence, out_type=int) + [EOS_ID]

    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

    tgt_ids = [BOS_ID]

    with torch.no_grad():

        for _ in range(max_len):

            tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(DEVICE)

            output = lstm_model(src_tensor, tgt_tensor)

            next_token = output[:, -1, :].argmax(dim=-1).item()

            if next_token == EOS_ID:
                break

            tgt_ids.append(next_token)

    if len(tgt_ids) == 1:
        return "Translation not found"

    return sp.decode(tgt_ids[1:])


# -------------------------------------------------------
# Common Translation Function
# -------------------------------------------------------

def translate_text(sentence, model_name):

    sentence = sentence.strip()

    if sentence == "":
        return "Please enter some text."

    try:

        if model_name.lower() == "lstm":
            return lstm_translate(sentence)

        return transformer_translate(sentence)

    except Exception as e:

        print("Translation Error:", e)

        return "Translation failed."
# -------------------------------------------------------
# HOME
# -------------------------------------------------------

@app.route("/")
def home():
    return redirect(url_for("login"))


# -------------------------------------------------------
# REGISTER
# -------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        print(request.form)   # <-- Add this line

        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        print(username, email, password)

        if not username or not email or not password:
            flash("Please fill all fields.", "danger")
            return redirect(url_for("register"))

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        existing_user = cursor.fetchone()

        if existing_user:
            flash("Email already registered!", "danger")
            conn.close()
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        cursor.execute(
            """
            INSERT INTO users(username,email,password,login_time)
            VALUES(?,?,?,?)
            """,
            (
                username,
                email,
                hashed_password,
                "First Login"
            )
        )

        conn.commit()
        conn.close()

        flash("Registration Successful! Please Login.", "success")

        return redirect(url_for("login"))

    return render_template("register.html")

# -------------------------------------------------------
# LOGIN
# -------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"].strip()
        password = request.form["password"]

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        user = cursor.fetchone()

        if user and check_password_hash(user["password"], password):

            session["user"] = user["username"]

            cursor.execute(
                """
                UPDATE users
                SET login_time=?
                WHERE username=?
                """,
                (
                    datetime.now().strftime("%d-%m-%Y %H:%M"),
                    user["username"]
                )
            )

            conn.commit()
            conn.close()

            flash(f"Welcome {user['username']}!", "success")

            return redirect(url_for("dashboard"))

        conn.close()

        flash("Invalid Email or Password!", "danger")

    return render_template("login.html")


# -------------------------------------------------------
# LOGOUT
# -------------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully.", "info")

    return redirect(url_for("login"))
# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    # Total translations by current user
    cursor.execute(
        "SELECT COUNT(*) FROM history WHERE username=?",
        (session["user"],)
    )
    total_translations = cursor.fetchone()[0]

    # Total registered users
    cursor.execute(
        "SELECT COUNT(*) FROM users"
    )
    total_users = cursor.fetchone()[0]

    # Last login
    cursor.execute(
        "SELECT login_time FROM users WHERE username=?",
        (session["user"],)
    )

    user = cursor.fetchone()

    conn.close()

    return render_template(
        "dashboard.html",
        username=session["user"],
        total_translations=total_translations,
        total_users=total_users,
        last_login=user["login_time"],
        metrics=metrics
    )
# -------------------------------------------------------
# TRANSLATOR
# -------------------------------------------------------

@app.route("/translator", methods=["GET", "POST"])
def translator():

    if "user" not in session:
        return redirect(url_for("login"))

    translation = ""
    input_text = ""
    selected_model = "Transformer"

    if request.method == "POST":

        input_text = request.form["inputText"]

        selected_model = request.form.get(
            "model",
            "Transformer"
        )

        translation = translate_text(
            input_text,
            selected_model
        )

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO history
            (username,model,input_text,translation)

            VALUES(?,?,?,?)
            """,

            (
                session["user"],
                selected_model,
                input_text,
                translation
            )
        )

        conn.commit()
        conn.close()

    return render_template(

        "index.html",

        username=session["user"],

        translation=translation,

        input_text=input_text,

        selected_model=selected_model

    )
# -------------------------------------------------------
# API TRANSLATION
# -------------------------------------------------------

@app.route("/translate", methods=["POST"])
def translate():

    if "user" not in session:
        return jsonify({"error": "Login Required"})

    data = request.get_json()

    text = data.get("text", "")

    model = data.get("model", "Transformer")

    translation = translate_text(
        text,
        model
    )

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history
        (username,model,input_text,translation)

        VALUES(?,?,?,?)
        """,

        (
            session["user"],
            model,
            text,
            translation
        )
    )

    conn.commit()
    conn.close()

    return jsonify({

        "translation": translation

    })
# -------------------------------------------------------
# HISTORY
# -------------------------------------------------------

@app.route("/history")
def history():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id,
               model,
               input_text,
               translation,
               translated_at
        FROM history
        WHERE username=?
        ORDER BY id DESC
    """, (session["user"],))

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )
# -------------------------------------------------------
# DELETE HISTORY
# -------------------------------------------------------

@app.route("/delete_history/<int:id>")
def delete_history(id):

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM history
        WHERE id=?
        """,
        (id,)
    )

    conn.commit()
    conn.close()

    flash("Translation deleted successfully.", "success")

    return redirect(url_for("history"))
# -------------------------------------------------------
# PROFILE
# -------------------------------------------------------

@app.route("/profile")
def profile():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE username=?
        """,
        (session["user"],)
    )

    user = cursor.fetchone()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM history
        WHERE username=?
        """,
        (session["user"],)
    )

    total = cursor.fetchone()[0]

    conn.close()

    return render_template(
        "profile.html",
        user=user,
        total=total
    )
# -------------------------------------------------------
# PERFORMANCE
# -------------------------------------------------------

@app.route("/performance")
def performance():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            SUM(CASE WHEN model='Transformer' THEN 1 ELSE 0 END),
            SUM(CASE WHEN model='LSTM' THEN 1 ELSE 0 END)
        FROM history
        WHERE username=?
    """, (session["user"],))

    usage = cursor.fetchone()

    conn.close()

    transformer_count = usage[0] if usage[0] else 0
    lstm_count = usage[1] if usage[1] else 0

    return render_template(
        "performance.html",
        metrics=metrics,
        transformer_count=transformer_count,
        lstm_count=lstm_count
    )
# -------------------------------------------------------
# ABOUT
# -------------------------------------------------------

@app.route("/about")
def about():

    return render_template("about.html")
# -------------------------------------------------------
# 404
# -------------------------------------------------------

@app.errorhandler(404)
def page_not_found(e):

    return render_template("404.html"),404
# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__=="__main__":

    app.run(debug=True)