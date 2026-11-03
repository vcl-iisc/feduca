from utils_libs import *
from utils_dataset import *
from utils_models import *
from config import cfg, process_args
import random


# Global parameters
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0")
max_norm = 10

def set_seed(seed: int, deterministic_cudnn: bool = True):
    """Seed Python, NumPy and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    if data_x.shape[0] == 0:
        breakpoint()
    batch_size = min(6000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
    
    loss_overall /= n_tst
    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall / n_tst

# --- Helper functions

def interpolate_test(tst_clnt_x, tst_clnt_y, model_func, local_model,global_model, dataset_name, n_par, w_decay = None):
    sweep = np.arange(0.0, 1.1, 0.1)
    temp_data_fit = np.zeros((len(sweep),)).astype(np.float32)
    temp_data_fit_loss = np.zeros((len(sweep),)).astype(np.float32)
    local_params = get_mdl_params([local_model], n_par)[0]
    gobal_params = get_mdl_params([global_model], n_par)[0]
    for clnt_weight in sweep:
        clnt_weight = round(clnt_weight, 1)
        index = int(clnt_weight/0.1)
        # evaluate the utility with sweeps on weights of the present model on the client
        mdl_params_to_assess = (1.0 - clnt_weight)*local_params + clnt_weight*gobal_params
        mdl_to_assess = model_func().to(device)
        mdl_to_assess = set_client_from_params(mdl_to_assess, mdl_params_to_assess)
        temp_data_fit_loss[index], temp_data_fit[index] = get_acc_loss(tst_clnt_x, tst_clnt_y, mdl_to_assess, dataset_name)
    #print("temp_data_fit:",temp_data_fit)    
    return  np.max(temp_data_fit)                

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    mdl.load_state_dict(dict_param)    
    return mdl



def set_mdl_grad(mdl, flat_grad):
    idx = 0
    for name, param in mdl.named_parameters():
        shape = param.data.shape
        numel = param.data.numel()
        
        grad_slice = flat_grad[idx: idx + numel]  
        grad_tensor = torch.tensor(grad_slice, dtype=param.data.dtype, device=param.data.device).reshape(shape)   
        param.grad = grad_tensor
        
        idx += numel
    
    return mdl



def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

# --- Train functions

def train_model(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    # print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    for e in range(epoch):
        # Training
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
            model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y, learning_rate, batch_size, n_minibatch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    n_iter_per_epoch = int(np.ceil(n_trn/batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
    count_step = 0
    is_done = False
    
    step_loss = 0; n_data_step = 0
    for e in range(epoch):
        # Training
        if is_done:
            break
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            count_step += 1
            if count_step > n_minibatch:
                is_done = True
                break
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = torch.sum(local_par_list * state_params_diff)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]

            if (count_step) % print_per == 0:
                step_loss /= n_data_step
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = get_mdl_params([model], n_par)
                    step_loss += (weight_decay)/2 * np.sum(params * params)
                print("Step %3d, Training Loss: %.4f" %(count_step, step_loss))
                step_loss = 0; n_data_step = 0
                model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_feddyn_mdl(model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model.train(); model = model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

###
def train_fedprox_mdl(model, avg_model_param_, mu, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    
    n_par = len(avg_model_param_)
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn/batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = mu/2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += weight_decay/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_feddisco_mdl(model, trn_x, trn_y,
                       learning_rate, batch_size, epoch,
                       print_per, weight_decay, dataset_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    # compute D_k once
    labels = trn_y.flatten()
    num_classes = cfg['n_class'] 
    counts = np.bincount(labels, minlength=num_classes)
    D_k = counts / counts.sum()
    
    # data loader
    trn_ds  = Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name)
    trn_gen = data.DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay or 0.0)
    
    n_trn = len(trn_x)
    for e in range(epoch):
        epoch_loss = 0.0
        for batch_x, batch_y in trn_gen:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            y_pred = model(batch_x)
            loss   = loss_fn(y_pred, batch_y.view(-1).long())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
        if (e+1) % print_per == 0:
            print(f"[Local] Epoch {e+1:3d}  Loss: {epoch_loss/n_trn:.4f}")
    
    # freeze & eval mode
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    return model, D_k

# def train_model(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
#     n_trn = trn_x.shape[0]
#     trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
#     # print(model)
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     model.train(); model = model.to(device)
#     for e in range(epoch):
#         # Training
        
#         trn_gen_iter = trn_gen.__iter__()
#         for i in range(int(np.ceil(n_trn/batch_size))):
#             batch_x, batch_y = trn_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             y_pred = model(batch_x)
#             loss = loss_fn(y_pred, batch_y.reshape(-1).long())
#             loss = loss / list(batch_y.size())[0]
            
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
#             optimizer.step()

#         if (e+1) % print_per == 0:
#             loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
#             print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
#             model.train()
        
#     # Freeze model
#     for params in model.parameters():
#         params.requires_grad = False
#     model.eval()
            
#     return model

def train_model_fedSAM(
    model, trn_x, trn_y,
    learning_rate, batch_size, epoch,
    print_per, weight_decay, dataset_name,
    rho=0.05, max_norm=10
):
    """
    Train local client model using FedSAM (Sharpness-Aware Minimization).

    Args:
        model: PyTorch model
        trn_x, trn_y: local training data
        learning_rate: initial lr
        batch_size: mini-batch size
        epoch: number of local epochs
        print_per: logging frequency
        weight_decay: L2 regularization
        dataset_name: dataset identifier for custom Dataset
        rho: SAM neighborhood radius
        max_norm: gradient clipping norm
    Returns:
        model: locally trained and frozen model
    """
    # prepare data loader
    trn_loader = data.DataLoader(
        Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
        batch_size=batch_size, shuffle=True
    )

    # use sum reduction + manual divide to match SAM paper
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    base_optim = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    model = model.to(device)
    model.train()

    for e in range(epoch):
        for batch_x, batch_y in trn_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # === STEP 1: clean gradient computation ===
            base_optim.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y.view(-1).long()) / batch_y.size(0)
            loss.backward()

            # compute explicit 2-norm of all gradients
            grad_norm = torch.norm(
                torch.stack([
                    p.grad.norm(p=2)
                    for p in model.parameters()
                    if p.grad is not None
                ])
            )

            # === STEP 2: ascent (perturbation) ===
            epsilon = []
            for p in model.parameters():
                if p.grad is None:
                    epsilon.append(torch.zeros_like(p))
                else:
                    # scale gradient to neighborhood radius
                    e_sam = p.grad * (rho / (grad_norm + 1e-12))
                    p.data.add_(e_sam)
                    epsilon.append(e_sam)

            # === STEP 3: gradient at perturbed weights ===
            base_optim.zero_grad()
            logits_adv = model(batch_x)
            loss_adv = loss_fn(logits_adv, batch_y.view(-1).long()) / batch_y.size(0)
            loss_adv.backward()

            # === STEP 4: restore & descent ===
            for p, e_sam in zip(model.parameters(), epsilon):
                p.data.sub_(e_sam)

            # clip and step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            base_optim.step()

        # logging
        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
            model.train()
    # freeze for safe evaluation
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def train_model_per_fedavg(model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, alpha=0.01, max_norm=1.0):
    """
    Per-FedAvg adapted training loop.
    learning_rate = Beta (outer meta-update step size)
    alpha = Inner local update step size
    """
    n_trn = trn_x.shape[0]
    
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # The SGD optimizer acts as the "Outer" meta-optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    model.train()
    model = model.to(device)
    
    for e in range(epoch):
        
        # Use an iterator to manually pull Support and Query pairs safely
        trn_gen_iter = iter(trn_gen)
        
        while True:
            # 1. Fetch Support Batch
            try:
                batch_x_sup, batch_y_sup = next(trn_gen_iter)
            except StopIteration:
                break # End of epoch
                
            # 2. Fetch Query Batch
            try:
                batch_x_qry, batch_y_qry = next(trn_gen_iter)
            except StopIteration:
                # Edge case: If odd number of batches, reuse support as query
                batch_x_qry, batch_y_qry = batch_x_sup, batch_y_sup
                
            batch_x_sup, batch_y_sup = batch_x_sup.to(device), batch_y_sup.to(device)
            batch_x_qry, batch_y_qry = batch_x_qry.to(device), batch_y_qry.to(device)

            # ==========================================
            # STEP 1: Backup original weights
            # ==========================================
            original_weights = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

            # ==========================================
            # STEP 2: Inner Update (on Support Batch)
            # ==========================================
            y_pred_sup = model(batch_x_sup)
            loss_sup = loss_fn(y_pred_sup, batch_y_sup.reshape(-1).long())
            loss_sup = loss_sup / batch_y_sup.size(0) 
            
            optimizer.zero_grad()
            loss_sup.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            
            # Manually apply the inner update (using alpha)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        p.data.sub_(alpha * p.grad.data)

            # ==========================================
            # STEP 3: Outer Update (on Query Batch)
            # ==========================================
            # Forward pass using the TEMPORARILY UPDATED weights
            y_pred_qry = model(batch_x_qry)
            loss_qry = loss_fn(y_pred_qry, batch_y_qry.reshape(-1).long())
            loss_qry = loss_qry / batch_y_qry.size(0)
            
            # Clear the inner gradients before computing the outer ones!
            optimizer.zero_grad() 
            loss_qry.backward() 
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            
            # ==========================================
            # STEP 4: Restore weights and apply Optimizer
            # ==========================================
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(original_weights[n])
            
            # Apply the outer meta-update to the ORIGINAL weights
            optimizer.step()

        # Logging
        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
            model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model