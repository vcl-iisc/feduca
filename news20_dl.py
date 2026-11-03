import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer

def load_glove_embeddings(path, word_to_idx, embedding_dim=300):
    print(f"Reading GloVe file from {path}...")
    embeddings_index = {}
    try:
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except FileNotFoundError:
        print(f"❌ Error: GloVe file not found at {path}. Returning random weights.")
        return torch.randn(len(word_to_idx), embedding_dim)

    vocab_size = len(word_to_idx)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    
    hits, misses = 0, 0
    for word, i in word_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
            
    print(f"✅ GloVe alignment: {hits} hits, {misses} misses (OOV).")
    return torch.from_numpy(embedding_matrix).float()


def get_strategic_indices_20news(labels, num_clients=10, num_classes=20, discovery_divisor=2, background_samples=3):
    """
    Strategic Split with Background Injection:
    - Experts get the vast majority of their 2 primary classes.
    - Everyone else gets a tiny injection (<5 samples) of the other 18 classes to prevent collapse.
    """
    labels = np.array(labels)
    train_indices = [[] for _ in range(num_clients)]
    val_indices = [[] for _ in range(num_clients)]
    
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    for c in range(num_classes):
        # Split pool into 80% Train, 20% Val
        indices = class_indices[c]
        split_pt = int(0.8 * len(indices))
        train_pool = indices[:split_pt]
        val_pool = indices[split_pt:]
        
        # 1. TRAINING DISTRIBUTION
        # Determine exactly which client is the "expert" for this class (since Client i has 2i and 2i+1)
        expert_client = c // 2  
        bg_pointer = 0
        
        for i in range(num_clients):
            if i != expert_client:
                # Inject a tiny number of background samples for the non-experts
                chunk = train_pool[bg_pointer : bg_pointer + background_samples]
                train_indices[i].extend(chunk)
                bg_pointer += len(chunk)
                
        # The expert gets everything that is left over in the training pool
        expert_chunk = train_pool[bg_pointer:]
        train_indices[expert_client].extend(expert_chunk)
            
        # 2. VALIDATION DISTRIBUTION (The Core Business + The Carrot)
        for i in range(num_clients):
            primary_1 = (2 * i) % num_classes
            primary_2 = (2 * i + 1) % num_classes
            discovery_1 = (2 * i + 2) % num_classes
            discovery_2 = (2 * i + 3) % num_classes
            
            if c in [primary_1, primary_2]:
                # Expert defends their core business
                val_indices[i].extend(val_pool)
            elif c in [discovery_1, discovery_2]:
                # The Shrunken Carrot to incentivize federation
                chunk_size = max(1, len(val_pool) // discovery_divisor) 
                val_indices[i].extend(val_pool[:chunk_size])
                
    return train_indices, val_indices


def get_strategic_indices_20news_old(labels, num_clients=10, num_classes=20, discovery_divisor=20):
    """
    Strictly Disjoint Sliding Window for 20 Newsgroups:
    10 Clients, 20 Classes -> Exactly 2 distinct Expert Classes per client.
    Train: [2i, 2i+1] (100% of available train pool)
    Val: [2i, 2i+1] (Heavy) + [2i+2, 2i+3] (Tiny fraction/Carrot)
    """
    labels = np.array(labels)
    train_indices = [[] for _ in range(num_clients)]
    val_indices = [[] for _ in range(num_clients)]
    
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    for c in range(num_classes):
        # Split pool into 80% Train, 20% Val
        indices = class_indices[c]
        split_pt = int(0.8 * len(indices))
        train_pool = indices[:split_pt]
        val_pool = indices[split_pt:]
        
        for i in range(num_clients):
            # Client i's designated classes
            primary_1 = (2 * i) % num_classes
            primary_2 = (2 * i + 1) % num_classes
            
            # The two classes they need to discover to incentivize federation
            discovery_1 = (2 * i + 2) % num_classes
            discovery_2 = (2 * i + 3) % num_classes
            
            # 1. TRAINING (Expertise)
            # Since classes are strictly disjoint, client gets the ENTIRE train pool for their classes
            if c in [primary_1, primary_2]:
                train_indices[i].extend(train_pool)
            
            # 2. VALIDATION: EXPERTISE (The Core Business)
            if c in [primary_1, primary_2]:
                val_indices[i].extend(val_pool)
                
            # 3. VALIDATION: DISCOVERY (The Shrunken Carrot)
            elif c in [discovery_1, discovery_2]:
                chunk = max(1, len(val_pool) // discovery_divisor) 
                val_indices[i].extend(val_pool[:chunk])
                
    return train_indices, val_indices


def save_federated_bundle(client_dls, test_dl, glove_matrix, filename="20news_federated_bundle.pt"):
    bundle = {
        'clients': {},
        'test_dataset': test_dl.dataset,
        'glove_matrix': glove_matrix,
        'metadata': {
            'num_clients': len(client_dls),
            'batch_size': test_dl.batch_size,
            'vocab_size': glove_matrix.shape[0] if glove_matrix is not None else 0,
            'embed_dim': glove_matrix.shape[1] if glove_matrix is not None else 0
        }
    }
    for client_id, loaders in client_dls.items():
        bundle['clients'][client_id] = {'train': loaders['train'].dataset, 'val': loaders['val'].dataset}
    torch.save(bundle, filename)
    print(f"💾 Bundle saved to {filename}")


def prepare_federated_20news(model_name="bert-base-uncased", num_clients=10, batch_size=16, max_length=128, glove_path=None):
    print("Loading 20 Newsgroups dataset via scikit-learn...")
    # Removing headers, footers, and quotes forces the model to learn semantics rather than metadata
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    all_texts = np.concatenate([newsgroups_train.data, newsgroups_test.data])
    all_labels = np.concatenate([newsgroups_train.target, newsgroups_test.target])

    combined_dataset = Dataset.from_dict({"text": all_texts, "labels": all_labels})
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    print("Tokenizing and formatting...")
    tokenized = combined_dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in ["input_ids", "attention_mask", "labels"]])
    tokenized.set_format("torch")
    
    # Global Test set (Reserve the last 10% of the entire dataset for server evaluation)
    total_size = len(tokenized)
    test_size = int(0.1 * total_size)
    train_val_ds = tokenized.select(range(total_size - test_size))
    test_ds = tokenized.select(range(total_size - test_size, total_size))
    
    # Create the Strategic Splits on the remaining 90%
    print(f"Generating strictly disjoint indices for {num_clients} clients...")
    train_indices, val_indices = get_strategic_indices_20news(np.array(train_val_ds["labels"]), num_clients, num_classes=20,discovery_divisor=3)
    
    client_dataloaders = {}
    for i in range(num_clients):
        client_dataloaders[f"client_{i}"] = {
            "train": DataLoader(train_val_ds.select(train_indices[i]), batch_size=batch_size, shuffle=True),
            "val": DataLoader(train_val_ds.select(val_indices[i]), batch_size=batch_size, shuffle=False)
        }
    
    global_test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    glove_matrix = None
    if glove_path:
        glove_matrix = load_glove_embeddings(glove_path, tokenizer.get_vocab(), embedding_dim=300)
    
    return client_dataloaders, global_test_dl, glove_matrix

if __name__ == "__main__":
    np.random.seed(1110)
    num_clients = 10
    GLOVE_FILE_PATH = "glove.6B.300d.txt" 
    
    client_dls, test_dl, glove_matrix = prepare_federated_20news(num_clients=num_clients, glove_path=GLOVE_FILE_PATH)
    
    print("\n--- Strategic Split Verification ---")
    for i in range(num_clients):
        c_id = f"client_{i}"
        t_labels = client_dls[c_id]["train"].dataset["labels"]
        v_labels = client_dls[c_id]["val"].dataset["labels"]
        
        t_counts = np.bincount(t_labels, minlength=20)
        v_counts = np.bincount(v_labels, minlength=20)
        
        print(f"\n{c_id.upper()}:")
        print(f"  TRAIN Expertise: {t_counts}") 
        print(f"  VAL Discovery:   {v_counts}")
        
        expert_classes = np.where(t_counts > 0)[0]
        val_classes = np.where(v_counts > 0)[0]
        discovery_class = set(val_classes) - set(expert_classes)
        
        print(f"  Expert in: {list(expert_classes)} | New Classes to learn: {list(discovery_class)}")

    save_federated_bundle(client_dls, test_dl, glove_matrix)