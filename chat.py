import os
import pickle
import torch
import re
from model import GPTConfig, GPT
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
ckpt_path = os.path.join('out', 'agm_3.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model = GPT(GPTConfig(**checkpoint['model_args']))
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
is_word_level = ' ' in itos.get(0, '') or ' ' in itos.get(1, '') or len(stoi) > 500
def encode(s):
    if is_word_level:
        res = []
        for w in re.findall(r"[\w']+|[^\w\s]", s):
            if w in stoi:
                res.append(stoi[w])
            elif w.lower() in stoi:
                res.append(stoi[w.lower()])
            else:
                for char in w:
                    if char in stoi:
                        res.append(stoi[char])
        return res
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    if is_word_level:
        s = ' '.join([itos[i] for i in l])
        s = s.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(" '", "'")
        s = s.replace('< | user | >', '<|user|>').replace('< | ai | >', '<|ai|>').replace('< | eos | >', '<|eos|>')
        return s
    return ''.join([itos[i] for i in l])

print(f"Chatting with AGM (type 'exit' to quit)")
eos_tokens = encode("<|eos|>")

while True:
    try:
        user_input = input("You: ")
    except EOFError:
        break
    if user_input.lower() == 'exit': break
    
    full_prompt = f"<|user|> {user_input} <|ai|>"
    prompt_ids = encode(full_prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
    
    generated_ids = []
    print("AGM: ", end="", flush=True)
    
    needs_space = False
    buffer = []
    stopped_by_eos = False
    in_code_block = False
    
    with torch.no_grad():
        for _ in range(300): 
            logits, _ = model(x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:])
            logits = logits[:, -1, :] / 0.7
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            next_val = next_id.item()
            generated_ids.append(next_val)
            x = torch.cat((x, next_id), dim=1)
            
            if generated_ids[-len(eos_tokens):] == eos_tokens:
                stopped_by_eos = True
                break
            
            buffer.append(next_val)
            if len(buffer) < len(eos_tokens):
                continue
                
            to_print_id = buffer.pop(0)
            token_text = itos[to_print_id]
            
            if is_word_level:
                
                if token_text == '`':
                    
                    print('`', end='', flush=True)
                    needs_space = False
                elif token_text in ".,!?;:":
                    print(token_text, end="", flush=True)
                    needs_space = True
                    if token_text == ':' and in_code_block:
                        print("\n    ", end="", flush=True)
                        needs_space = False
                elif token_text in "()[]{}":
                    print(token_text, end="", flush=True)
                    needs_space = False
                elif token_text in '"\'':
                    print(token_text, end="", flush=True)
                    needs_space = False
                else:
                    if needs_space: print(" ", end="")
                    print(token_text, end="", flush=True)
                    needs_space = True
                    
                
                if '```' in itos[to_print_id]: 
                    print("\n", end="")
                    in_code_block = not in_code_block
            else:
                print(token_text, end="", flush=True)
        
        if not stopped_by_eos:
            for to_print_id in buffer:
                token_text = itos[to_print_id]
                print(token_text, end="", flush=True)
    
    print("\n" + "-"*30 + "\n")
