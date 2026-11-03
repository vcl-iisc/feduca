import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
#from datasets import load_dataset, concatenate_datasets


import torch
import os

def save_federated_bundle(client_dls, test_dl, filename="sst5_federated_bundle.pt"):
    """
    Saves all partitioned datasets and the global test set into one file.
    """
    # Create the dictionary structure
    bundle = {
        'clients': {},
        'test_dataset': test_dl.dataset,
        'metadata': {
            'num_clients': len(client_dls),
            'batch_size': test_dl.batch_size
        }
    }
    
    # Extract datasets from the client loaders
    for client_id, loaders in client_dls.items():
        bundle['clients'][client_id] = {
            'train': loaders['train'].dataset,
            'val': loaders['val'].dataset
        }
    
    torch.save(bundle, filename)
    print(f"✅ All 11 DataLoaders (5 Train, 5 Val, 1 Test) saved to {filename}")




def get_dirichlet_client_indices(labels, num_clients, alpha):
    """
    Distributes dataset indices to clients enforcing equal sizes and a Dirichlet label skew.
    """
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    n_samples = len(labels)
    
    # Calculate exact quota to ensure equal sizes (drop tiny remainder if any)
    quota = n_samples // num_clients 
    client_indices = [[] for _ in range(num_clients)]
    
    # Generate Dirichlet proportions for each class across clients
    class_props = {
        c: np.random.dirichlet([alpha] * num_clients) 
        for c in range(num_classes)
    }
    
    # Group indices by class and shuffle
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])
        
    # Assign items sequentially based on Dirichlet weights, enforcing quota limits
    for c, indices in class_indices.items():
        for idx in indices:
            weights = class_props[c].copy()
            
            # Mask out clients that have reached their equal-size quota
            for i in range(num_clients):
                if len(client_indices[i]) >= quota:
                    weights[i] = 0.0
                    
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # Fallback: if preferred clients are full, pick uniformly from remaining
                weights = np.array([1.0 if len(client_indices[i]) < quota else 0.0 for i in range(num_clients)])
                if weights.sum() == 0:
                    continue # Discard remainders to ensure perfectly equal sizes
                weights = weights / weights.sum()
                
            chosen_client = np.random.choice(num_clients, p=weights)
            client_indices[chosen_client].append(idx)
            
    return client_indices


def prepare_federated_sst5(model_name="bert-base-uncased", num_clients=5, batch_size=16, max_length=128):
    print("Loading SST-5 (5-class version)...")
    # Using setfit/sst5 which is a very clean 5-class version
    dataset = load_dataset("setfit/sst5")
    
    # Combine all splits for custom 70/20/10 redistribution
    combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # SST-5 uses "text", SST-2 uses "sentence"
        column = "text" if "text" in examples else "sentence"
        return tokenizer(examples[column], padding="max_length", truncation=True, max_length=max_length)

    print("Tokenizing data...")
    tokenized = combined_dataset.map(tokenize_function, batched=True)
    
    # Clean up columns. SST-5 might have 'label_text', 'text', etc.
    # We only want to keep what the model needs.
    current_columns = tokenized.column_names
    keep_columns = ["input_ids", "attention_mask", "label"]
    columns_to_remove = [c for c in current_columns if c not in keep_columns]
    
    tokenized = tokenized.remove_columns(columns_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    
    # Split 70/20/10
    total_len = len(tokenized)
    train_len = int(0.70 * total_len)
    val_len = int(0.20 * total_len)
    
    tokenized = tokenized.shuffle(seed=42)
    train_ds = tokenized.select(range(0, train_len))
    val_ds = tokenized.select(range(train_len, train_len + val_len))
    test_ds = tokenized.select(range(train_len + val_len, total_len))
    
    # Distribute among 5 clients
    import numpy as np
    print(f"Distributing 5-class data (Train alpha=0.5, Val alpha=1.0)...")
    train_client_indices = get_dirichlet_client_indices(np.array(train_ds["labels"]), num_clients, alpha=0.5)
    val_client_indices = get_dirichlet_client_indices(np.array(val_ds["labels"]), num_clients, alpha=1.0)
    
    client_dataloaders = {}
    for i in range(num_clients):
        c_train = train_ds.select(train_client_indices[i])
        c_val = val_ds.select(val_client_indices[i])
        
        client_dataloaders[f"client_{i}"] = {
            "train": DataLoader(c_train, batch_size=batch_size, shuffle=True),
            "val": DataLoader(c_val, batch_size=batch_size, shuffle=False)
        }
        
    global_test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return client_dataloaders, global_test_dataloader


def prepare_federated_sst2(model_name="bert-base-uncased", num_clients=5, batch_size=16, max_length=128):
    print("Loading and preparing SST-2 dataset...")
    # 1. Load and combine dataset (GLUE test set lacks labels, so we combine train+val)
    dataset = load_dataset("glue", "sst2")
    combined_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
    
    # 2. Tokenization and Formatting
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

    print("Tokenizing data...")
    tokenized = combined_dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    
    # 3. Global 70/20/10 Split
    total_len = len(tokenized)
    train_len = int(0.70 * total_len)
    val_len = int(0.20 * total_len)
    test_len = total_len - train_len - val_len
    
    # Shuffle before splitting globally
    tokenized = tokenized.shuffle(seed=42)
    train_ds = tokenized.select(range(0, train_len))
    val_ds = tokenized.select(range(train_len, train_len + val_len))
    test_ds = tokenized.select(range(train_len + val_len, total_len))
    
    # 4. Distribute Data to Clients
    print(f"Distributing Train Data (alpha=0.5) to {num_clients} clients...")
    train_client_indices = get_dirichlet_client_indices(train_ds["labels"], num_clients, alpha=0.005)
    
    print(f"Distributing Validation Data (alpha=1.0) to {num_clients} clients...")
    val_client_indices = get_dirichlet_client_indices(val_ds["labels"], num_clients, alpha=1.0)
    
    # 5. Build DataLoaders
    client_dataloaders = {}
    for i in range(num_clients):
        # Extract subsets for each client
        client_train_ds = train_ds.select(train_client_indices[i])
        client_val_ds = val_ds.select(val_client_indices[i])
        
        client_dataloaders[f"client_{i}"] = {
            "train": DataLoader(client_train_ds, batch_size=batch_size, shuffle=True),
            "val": DataLoader(client_val_ds, batch_size=batch_size, shuffle=False)
        }
        
    global_test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return client_dataloaders, global_test_dataloader

# --- Execution and Verification ---
if __name__ == "__main__":
    np.random.seed(2) # For reproducibility 
    
    client_dls, test_dl = prepare_federated_sst5(num_clients=5)
    
    print("\n--- Partition Verification ---")
    print(f"Global Test Set Dataloader: {len(test_dl.dataset)} points")
    
    for client_id, dls in client_dls.items():
        train_labels = dls["train"].dataset["labels"]
        val_labels = dls["val"].dataset["labels"]
        
        train_counts = np.bincount(train_labels, minlength=2)
        val_counts = np.bincount(val_labels, minlength=2)
        
        print(f"\n{client_id.upper()}:")
        print(f"  Train Set Size: {len(train_labels)} | Label Split (0/1): {train_counts}")
        print(f"  Val Set Size:   {len(val_labels)} | Label Split (0/1): {val_counts}")

    save_federated_bundle(client_dls, test_dl)    