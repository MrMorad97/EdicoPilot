import os
import pickle
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bert_score import score

from LM import LightweightTransformerChatbot, DailyDialogTokenizer, ConversationContext, generate_response
from fineTunedGpt2 import FinetunedGPT2Chat

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load pretrained GPT2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

# Load fine-tuned GPT2 wrapper
ft_gpt2_model = FinetunedGPT2Chat("./fine_tuned_gpt2")

# Load custom LM
with open("dailydialog_tokenizer.pkl", "rb") as f:
    lm_tokenizer = pickle.load(f)

custom_lm_model = LightweightTransformerChatbot(
    vocab_size=lm_tokenizer.vocab_size(),
    d_model=192,
    n_heads=8,
    n_layers=4,
    d_ff=512,
    max_len=128,
    dropout=0.1,
    pad_token_id=lm_tokenizer.pad_token_id
)
custom_lm_model.load_state_dict(torch.load("best_dailydialog_chatbot.pth", map_location=device))
custom_lm_model.eval()

def generate_response_gpt2(prompt: str, max_length=50):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(
        inputs,
        max_length=inputs.shape[1] + max_length,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        eos_token_id=gpt2_tokenizer.eos_token_id,
        pad_token_id=gpt2_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
    )
    generated = outputs[0][inputs.shape[1]:]
    return gpt2_tokenizer.decode(generated, skip_special_tokens=True).strip()

def generate_response_finetuned_gpt2(prompt: str):
    output = ""
    for chunk in ft_gpt2_model.stream_response(prompt):
        output += chunk
    return output.strip()

def generate_response_custom_lm(prompt: str):
    tokens = lm_tokenizer.encode(prompt)
    if len(tokens) > 128:
        tokens = tokens[-128:]
    truncated_prompt = lm_tokenizer.decode(tokens)
    temp_context = ConversationContext(lm_tokenizer)
    temp_context.add_utterance('user', truncated_prompt)
    return str(generate_response(custom_lm_model, lm_tokenizer, temp_context, "")).strip()

def clean_text(text):
    return text.strip().lower()

def compute_bertscore(pred, ref):
    P, R, F1 = score([pred], [ref], lang="en", device=device)
    return F1.item()

def main():
    dialogue_file = "dialogues_test.txt"

    with open(dialogue_file, "r", encoding="utf-8") as f:
        dialogue_line = f.readline()

    turns = [t.strip() for t in dialogue_line.split("__eou__") if t.strip()]
    print(f"Total turns in dialogue: {len(turns)}")

    for i in range(1, len(turns)):
        prompt = " __eou__ ".join(turns[:i]) + " __eou__"
        reference = turns[i]

        print(f"\n--- Turn {i} ---")
        print("Prompt:")
        print(prompt)
        print("Reference:")
        print(reference)

        try:
            pred_gpt2 = generate_response_gpt2(prompt)
            pred_ft_gpt2 = generate_response_finetuned_gpt2(prompt)
            pred_custom_lm = generate_response_custom_lm(prompt)

            # Clean for scoring
            ref_c = clean_text(reference)
            p1 = clean_text(pred_gpt2)
            p2 = clean_text(pred_ft_gpt2)
            p3 = clean_text(pred_custom_lm)

            # Compute BERTScore for each
            f1_gpt2 = compute_bertscore(p1, ref_c)
            f1_ft = compute_bertscore(p2, ref_c)
            f1_custom = compute_bertscore(p3, ref_c)

            print("\nPredictions:")
            print(f"[GPT-2 pretrained]       {pred_gpt2}")
            print(f"[Fine-tuned GPT-2]       {pred_ft_gpt2}")
            print(f"[Custom LM]              {pred_custom_lm}")

            print("\nBERTScore F1s:")
            print(f"GPT2:          {f1_gpt2:.4f}")
            print(f"FT GPT2:       {f1_ft:.4f}")
            print(f"Custom LM:     {f1_custom:.4f}")

        except Exception as e:
            print(f"Error on turn {i}: {e}")
            continue

if __name__ == "__main__":
    main()
