import pickle
import sys
import os


from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nanochat.dataset import parquets_iter_batched

with open('/local/s4859049/nanochat_cache/tokenizer_32k/tokenizer.pkl', 'rb') as f:
    tok = pickle.load(f)

ranks = tok._mergeable_ranks

def find_split(token_bytes):
    target_rank = ranks.get(token_bytes, None)
    
    valid_splits = []
    
    if target_rank is None or target_rank < 256 or len(token_bytes) <= 1:
        return None
    
    for i in range(1, len(token_bytes)):
        left = token_bytes[:i]
        right = token_bytes[i:]
        
        if left in ranks and right in ranks:
            rank_l = ranks[left]
            rank_r = ranks[right]
            
            if rank_l < target_rank and rank_r < target_rank:
                max_r = max(rank_l, rank_r) #The last merging will have higher rank
                valid_splits.append((max_r, left, right))
                
    if not valid_splits:
        return None
    valid_splits.sort()
    return valid_splits[0][1], valid_splits[0][2]

counts_cache = {}
def count_in_corpus(target_bytes):
    if target_bytes in counts_cache:
        return counts_cache[target_bytes]
    c = 0
    docs = 0
    for batch in parquets_iter_batched(split='train'):
        for doc in batch:
            c += doc.encode('utf-8').count(target_bytes) #calculate number of target byte occurs in the document
            docs += 1

    counts_cache[target_bytes] = c
    return c

def print_tree(token_bytes, prefix='', is_left = True):
    r = ranks.get(token_bytes, 'N/A')
    
    if r != 'N/A' and r >= 256:
        c = count_in_corpus(token_bytes) #The frequency of the token in documents
        count_str = f" | Count: {c:,}"
    else:
        count_str = " | Base Byte"
        
    label = f"{token_bytes} (Rank: {r}{count_str})" #Display token, rank, and count
    print(prefix + ('├── ' if is_left else '└── ') + label)
    
    split = find_split(token_bytes)
    if split:
        left, right = split
        new_prefix = prefix + ('│   ' if is_left else '    ')
        print_tree(left, new_prefix, True)
        print_tree(right, new_prefix, False)
        
print_tree(b' University', prefix='', is_left=False)
    
    

