import os
import sys
import pickle

# Setup path root nanochat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from nanochat.dataset import parquets_iter_batched

CACHE_DIR = "/local/AgenticLLM-Assignment1/nanochat_cache"
PATH_8K = os.path.join(CACHE_DIR, "tokenizer_8k/tokenizer.pkl")
PATH_32K = os.path.join(CACHE_DIR, "tokenizer_32k/tokenizer.pkl")

print("Loading tokenizer 8K dan 32K...")
with open(PATH_8K, 'rb') as f:
    tok_8k = pickle.load(f)

with open(PATH_32K, 'rb') as f:
    tok_32k = pickle.load(f)

sample_text = ( "Lorem Ipsum is simply dummy text of the printing and typesetting industry. " 
                "Lorem Ipsum has been the industry's standard dummy text ever since 1966, when designers at Letraset and James Mosley, " 
                "the librarian at St Bride Printing Library in London" )
            
            
ids_8k = tok_8k.encode(sample_text)
ids_32k = tok_32k.encode(sample_text)

tokens_8k = len(ids_8k)
tokens_32k = len(ids_32k)

tok_8k_list = []
tok_32k_list = []

for token in ids_8k:
    tok_8k_list.append(tok_8k.decode([token]))
for token in ids_32k:
    tok_32k_list.append(tok_32k.decode([token]))
    
print(f"8K: {tok_8k_list}")
print(f"32K: {tok_32k_list}")
print(tokens_8k, tokens_32k)

# Calculation
cr_8k = tokens_8k/len(sample_text)
cr_32k = tokens_32k/len(sample_text)
avg_len_8k = tokens_8k/1
avg_len_32k = tokens_32k /1
reduction_pct = ((tokens_8k - tokens_32k) / tokens_8k) * 100

print(f"\n{'='*45}")
print(f"EVALUATION RESULTS ({len(sample_text):,} bytes)")
print("=" * 45)
print(f"8K  : {tokens_8k:,} tokens | {cr_8k:.4f} token/chars | {avg_len_8k:.1f} token/docs")
print(f"32K : {tokens_32k:,} tokens | {cr_32k:.4f} token/chars | {avg_len_32k:.1f} token/docs")
print("-" * 45)
print(f"Token Reduction (32K vs 8K): {reduction_pct:.2f}%")
print("=" * 45)


#----------------------- 1.4 -----------------------------------
#---------------------------------------------------------------

text = "1+2=3 123 1234 12$1,250.00 vs -$500 3.14159 year2019"
print("8K :", [tok_8k.decode([t]) for t in tok_8k.encode(text)])
print("32K:", [tok_32k.decode([t]) for t in tok_32k.encode(text)])

code = "    for i in range(10):\n        return x != y"
print("8K :", [tok_8k.decode([t]) for t in tok_8k.encode(code)])
print("32K:", [tok_32k.decode([t]) for t in tok_32k.encode(code)])

non_eng = "Café / 你好世界 / naïve"
print("8K :", [tok_8k.decode([t]) for t in tok_8k.encode(non_eng)])
print("32K:", [tok_32k.decode([t]) for t in tok_32k.encode(non_eng)])
