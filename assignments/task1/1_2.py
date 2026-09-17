import os
import sys
import pickle

# Setup path root nanochat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from nanochat.dataset import parquets_iter_batched

CACHE_DIR = "/local/s4859049/nanochat_cache"
PATH_8K = os.path.join(CACHE_DIR, "tokenizer_8k/tokenizer.pkl")
PATH_32K = os.path.join(CACHE_DIR, "tokenizer_32k/tokenizer.pkl")

print("Loading tokenizer 8K dan 32K...")
with open(PATH_8K, 'rb') as f:
    tok_8k = pickle.load(f)

with open(PATH_32K, 'rb') as f:
    tok_32k = pickle.load(f)


num_chars = 0
tokens_8k = 0
tokens_32k = 0
doc_count = 0

for batch in parquets_iter_batched(split='val'):
    for doc in batch:
        text = doc.encode('utf-8')
        num_chars += len(text)
        
        # Encode the documents
        ids_8k = tok_8k.encode(doc)
        ids_32k = tok_32k.encode(doc)
        
        tokens_8k += len(ids_8k)
        tokens_32k += len(ids_32k)
        
        doc_count += 1
    break

text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, the librarian at St Bride Printing Library in London"
_8k = tok_8k.encode(text)
_32k = tok_32k.encode(text)

_tokens_8k = len(_8k)
_tokens_32k = len(_32k)

tok_8k_list = []
tok_32k_list = []
for token in _8k:
    tok_8k_list.append(tok_8k.decode([token]))
for token in _32k:
    tok_32k_list.append(tok_32k.decode([token]))
print(tok_8k_list)
print(tok_32k_list)
print(_tokens_8k, _tokens_32k)

# Calculation
cr_8k = tokens_8k/num_chars
cr_32k = tokens_32k/num_chars
avg_len_8k = tokens_8k / doc_count
avg_len_32k = tokens_32k / doc_count
reduction_pct = ((tokens_8k - tokens_32k) / tokens_8k) * 100

print(f"\n{'='*45}")
print(f"EVALUATION RESULTS ({doc_count:,} docs | {num_chars:,} bytes)")
print("=" * 45)
print(f"8K  : {tokens_8k:,} tokens | {cr_8k:.4f} token/chars | {avg_len_8k:.1f} token/docs")
print(f"32K : {tokens_32k:,} tokens | {cr_32k:.4f} token/chars | {avg_len_32k:.1f} token/docs")
print("-" * 45)
print(f"Token Reduction (32K vs 8K): {reduction_pct:.2f}%")
print("=" * 45)

text = "123456 2024 2025 3.14159"
print("8K :", [tok_8k.decode([t]) for t in tok_8k.encode(text)])
print("32K:", [tok_32k.decode([t]) for t in tok_32k.encode(text)])

code = "    for i in range(10):\n        return x != y"
print("8K :", [tok_8k.decode([t]) for t in tok_8k.encode(code)])
print("32K:", [tok_32k.decode([t]) for t in tok_32k.encode(code)])

non_eng = "Halo apa kabar? / 你好世界 / Goedemorgen"
print("8K :", [tok_8k.decode([t]) for t in tok_8k.encode(non_eng)])
print("32K:", [tok_32k.decode([t]) for t in tok_32k.encode(non_eng)])