import torch
import pandas as pd
import sentencepiece as spm
import sacrebleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.transformer import KharadaTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Tokenizer
# -----------------------

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/bpe.model")

pad_id = sp.pad_id()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
vocab_size = sp.get_piece_size()

# -----------------------
# Load Transformer
# -----------------------

model = KharadaTransformer(vocab_size=vocab_size).to(device)

model.load_state_dict(
    torch.load("model/best_model.pt", map_location=device)
)

model.eval()

# -----------------------
# Translation Function
# -----------------------

def translate(sentence, max_len=40):

    src_ids = [bos_id] + sp.encode(sentence, out_type=int) + [eos_id]

    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    tgt_ids = [bos_id]

    for _ in range(max_len):

        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(src_tensor, tgt_tensor, pad_id)

        next_token = output[:, -1, :].argmax(dim=-1).item()

        tgt_ids.append(next_token)

        if next_token == eos_id:
            break

    output_ids = tgt_ids[1:]

    if eos_id in output_ids:
        output_ids = output_ids[:output_ids.index(eos_id)]

    return sp.decode(output_ids)

# -----------------------
# Load Test Data
# -----------------------

df = pd.read_csv("data/test.csv")

predictions = []
references = []

print("Evaluating...\n")

for _, row in df.iterrows():

    prediction = translate(str(row["kharada"]))

    predictions.append(prediction)

    references.append(str(row["english"]))

# -----------------------
# BLEU
# -----------------------

bleu = sacrebleu.corpus_bleu(
    predictions,
    [references]
)

print("--------------------------------")
print("Transformer Evaluation")
print("--------------------------------")

print("BLEU Score :", bleu.score)

print("\nSample Predictions\n")

for i in range(min(10, len(predictions))):

    print("Input      :", df.iloc[i]["kharada"])
    print("Expected   :", references[i])
    print("Predicted  :", predictions[i])
    print("--------------------------------")

    # -----------------------
# Accuracy, Precision, Recall, F1
# -----------------------

true_words = []
pred_words = []

for ref, pred in zip(references, predictions):

    ref = ref.lower().split()
    pred = pred.lower().split()

    max_len = max(len(ref), len(pred))

    ref += ["<pad>"] * (max_len - len(ref))
    pred += ["<pad>"] * (max_len - len(pred))

    true_words.extend(ref)
    pred_words.extend(pred)

accuracy = accuracy_score(true_words, pred_words)

precision = precision_score(
    true_words,
    pred_words,
    average="micro",
    zero_division=0
)

recall = recall_score(
    true_words,
    pred_words,
    average="micro",
    zero_division=0
)

f1 = f1_score(
    true_words,
    pred_words,
    average="micro",
    zero_division=0
)

print("\n==============================")
print("Performance Metrics")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")