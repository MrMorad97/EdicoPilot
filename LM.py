import torch
import pickle
from collections import deque
import re

class DailyDialogTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.sep_token = '<SEP>'
        self.user_token = '<USER>'
        self.bot_token = '<BOT>'
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.start_token_id = 2
        self.end_token_id = 3
        self.sep_token_id = 4
        self.user_token_id = 5
        self.bot_token_id = 6

    def tokenize(self, text):
        text = text.lower()
        contractions = {
            r"won't": "will not", r"can't": "can not", r"n't": " not",
            r"'re": " are", r"'s": " is", r"'d": " would", r"'ll": " will",
            r"'t": " not", r"'ve": " have", r"'m": " am"
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)
        return re.findall(r'\b\w+\b|[^\w\s]', text)

    def encode(self, text):
        words = self.tokenize(text)
        return [self.word_to_id.get(word, self.unk_token_id) for word in words]

    def decode(self, ids):
        words = []
        for id in ids:
            if id == self.end_token_id:
                break
            if id not in [self.pad_token_id, self.start_token_id, self.sep_token_id, 
                         self.user_token_id, self.bot_token_id]:
                words.append(self.id_to_word.get(id, self.unk_token))
        return ' '.join(words)

    def vocab_size(self):
        return len(self.word_to_id)

class LightweightTransformerChatbot(torch.nn.Module):
    def __init__(self, vocab_size, d_model=192, n_heads=8, n_layers=4, 
                 d_ff=512, max_len=128, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.PositionalEncoding(d_model, max_len)
        self.transformer_blocks = torch.nn.ModuleList([
            self.TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.output_projection = torch.nn.Linear(d_model, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
        
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.output_projection.weight)
        self.output_projection.bias.data.zero_()

    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    class TransformerBlock(torch.nn.Module):
        def __init__(self, d_model, n_heads, d_ff, dropout):
            super().__init__()
            self.attention = self.MultiHeadAttention(d_model, n_heads, dropout)
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_ff),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(d_ff, d_model),
                torch.nn.Dropout(dropout)
            )
            self.dropout = torch.nn.Dropout(dropout)

        class MultiHeadAttention(torch.nn.Module):
            def __init__(self, d_model, n_heads, dropout):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_k = d_model // n_heads
                self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
                self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
                self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
                self.W_o = torch.nn.Linear(d_model, d_model)
                self.dropout = torch.nn.Dropout(dropout)
                self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

            def forward(self, query, key, value, mask=None):
                batch_size = query.size(0)
                Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, float('-inf'))
                attention_weights = torch.nn.functional.softmax(scores, dim=-1)
                attention_weights = self.dropout(attention_weights)
                context = torch.matmul(attention_weights, V)
                context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
                return self.W_o(context)

        def forward(self, x, mask=None):
            attn_output = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.ff(x)
            return self.norm2(x + ff_output)

    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()
        padding_mask = (x != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)
        mask = padding_mask & causal_mask.unsqueeze(0)
        
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
            
        logits = self.output_projection(x)
        return logits

class ConversationContext:
    def __init__(self, tokenizer, max_turns=3):
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns*2)
        
    def add_utterance(self, speaker, text):
        tokens = self.tokenizer.encode(text)
        self.history.append((speaker, tokens))
        
    def get_context_sequence(self):
        sequence = []
        for speaker, tokens in self.history:
            if speaker == 'user':
                sequence.append(self.tokenizer.user_token_id)
            else:
                sequence.append(self.tokenizer.bot_token_id)
            sequence.extend(tokens)
        return sequence

def generate_response(model, tokenizer, context, user_input, max_length=50, temperature=0.7, top_k=50):
    model.eval()
    device = next(model.parameters()).device
    context.add_utterance('user', user_input)
    
    input_seq = [tokenizer.start_token_id] 
    input_seq += context.get_context_sequence()
    input_seq += [tokenizer.sep_token_id]
    
    generated_ids = input_seq.copy()
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor([generated_ids], dtype=torch.long).to(device)
            logits = model(inputs)
            
            next_token_logits = logits[0, -1, :] / temperature
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                next_token_logits.scatter_(0, top_k_indices, top_k_logits)
            
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == tokenizer.end_token_id:
                break
            generated_ids.append(next_token)
    
    response_ids = []
    in_response = False
    for token in generated_ids:
        if token == tokenizer.sep_token_id:
            in_response = True
        elif in_response:
            if token == tokenizer.end_token_id:
                break
            response_ids.append(token)
    
    response = tokenizer.decode(response_ids)
    context.add_utterance('bot', response)
    return response

def load_model_and_tokenizer(model_path, tokenizer_path):
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Initialize model
    model = LightweightTransformerChatbot(
        vocab_size=tokenizer.vocab_size(),
        d_model=192,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        max_len=128,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, tokenizer

def chat_interactively(model, tokenizer):
    context = ConversationContext(tokenizer, max_turns=3)
    print("\nChat with the bot (type 'exit' to end):")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye! Have a great day!")
            break
            
        response = generate_response(
            model, 
            tokenizer, 
            context, 
            user_input,
            max_length=50,
            temperature=0.7,
            top_k=50
        )
        print(f"Bot: {response}")

if __name__ == "__main__":
    # Paths to your saved model and tokenizer
    model_path = "best_dailydialog_chatbot.pth"  # Update with your path
    tokenizer_path = "dailydialog_tokenizer.pkl"  # Update with your path
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    # Start interactive chat
    chat_interactively(model, tokenizer)