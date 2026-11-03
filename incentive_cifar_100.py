from utils_general import *
from utils_methods import *

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

# Also see which device is currently active:
print("Current device index:", torch.cuda.current_device())
# Dataset initialization

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

#     - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
#########


# Generate IID or Dirichlet distribution
# IID
n_client = 20 # setting number of clients to 20
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='iid', unbalanced_sgm=0)
data_obj = DatasetObject(dataset='CIFAR100', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.35, rule_arg='0.5', rule_arg_test='1.0', isolated_clnt='False')

###
model_name         = 'Resnet18' # Model type
com_amount         = 100 # Number of communication rounds
save_period        = 5 # frequency of saving the model to disk; save_instances = com_amount/save_period
weight_decay       = 1e-3
batch_size         = 50
act_prob           = 1 # probability of client being active in a communication round
lr_decay_per_round = 1
epoch              = 5 # client local training
learning_rate      = 0.1
print_per          = 1 # frequency of logging training updates 

# Model function
model_func = lambda : client_model(model_name)
# init_model = model_func()
# Initalise the model for all methods or load it from a saved initial model
init_model = model_func()
if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    
    

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)            # all GPUs
        # for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
dir_param = 0.5


ws_rounds = -1
print("Rational FedAvg")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed,ws_rounds = ws_rounds)


print("Rational FedProx")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_Rational_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,mu = 1e-4,lr_decay_per_round=lr_decay_per_round,rand_seed = seed,ws_rounds = ws_rounds)
#
print("Rational IncFL")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,rwpr, tst_perf_clnts] = train_rational_IncFL(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed,ws_rounds = ws_rounds)


print("Rational agggFFs")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_AggFFs(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed,ws_rounds = ws_rounds)
   

ws_rounds = 4
print("FedAvg ws 4  EXPERIMENT")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed,ws_rounds = ws_rounds)

print("UCA EXPERIMENT")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,
 rwpr, tst_perf_clnts] = train_feduca(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, 
                                      print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round,rand_seed = seed,dir_param=dir_param_,num_strategies=strat,extra_thr = 0.0,val_ratio=0.0)


#seed = 20
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)          
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#
#
#print("SECOND EXP")
#[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,
# rwpr, tst_perf_clnts] = train_lottery_nips_reb_persuasion_FedAvg_util(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
#                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                     model_func=model_func, init_model=init_model, save_period=save_period,
#                                     lr_decay_per_round=lr_decay_per_round,rand_seed = seed,dir_param=0.5,num_strategies=1)
#


#print("SECOND EXP")
#seed = 20
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)   
#[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,
# rwpr, tst_perf_clnts] = train_lottery_nips_reb_persuasion_FedAvg_util(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
#                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
#                                     model_func=model_func, init_model=init_model, save_period=save_period,
#                                     lr_decay_per_round=lr_decay_per_round,rand_seed = seed,dir_param=0.01)    
#


# # Plot results
# plt.figure(figsize=(6, 5))
# plt.plot(np.arange(com_amount)+1, tst_perf_sel_rational_FedAvg[:,1], label='rational game EPO')
# plt.ylabel('Test Accuracy', fontsize=16)
# plt.xlabel('Communication Rounds', fontsize=16)
# plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
# plt.xlim([0, com_amount+1])
# plt.title(data_obj.name, fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.savefig('Output/%s/plot.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')

# # Plotting client performance on validation set over communication rounds
# n_clients, n_rounds, _ = tst_perf_clnts.shape
# fig, axes = plt.subplots(n_clients, 1, figsize=(10, 20), sharex=True)   
# for client in range(n_clients):
#     axes[client].plot(np.arange(n_rounds), tst_perf_clnts[client, :, 1], marker='o', linestyle='-')
#     axes[client].set_title(f"Client {client+1}")
#     axes[client].set_ylabel("Accuracy")
#     axes[client].grid(True, linestyle='--', alpha=0.7)

# axes[-1].set_xlabel("Communication Round")
# fig.suptitle("Test Accuracy across Communication Rounds for All Clients", fontsize=16, y=0.98)

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.savefig('Output/%s/clnt_accuracy_plots.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.close(fig)

# # Plotting the round wise participation rate of clients
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(n_rounds), rwpr, marker='o', linestyle='-', color='blue')
# plt.title("RWPR over Communication Rounds")
# plt.xlabel("Communication Round")
# plt.ylabel("RWPR")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig('Output/%s/rwpr.pdf' %data_obj.name, dpi=1000, bbox_inches='tight')
# plt.close(fig)
