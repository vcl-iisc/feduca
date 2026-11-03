from utils_general import *
from utils_methods import *
import argparse
import torch.backends.cudnn as cudnn
from config import cfg, process_args

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    if k == 'control_name':
        continue
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
# args['contral_name']
args = vars(parser.parse_args())
process_args(args)


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
########tmux
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
n_client = cfg['n_clnts'] # setting number of clients to 10
# if cfg['iid']:
#     data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='iid', unbalanced_sgm=0)
# else:
#     data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.0, rule_arg='0.2', rule_arg_test='1.0', isolated_clnt='False')
#data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.0, rule_arg='1.0', rule_arg_test='1.0', isolated_clnt='False')

data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.0, rule_arg='0.5', rule_arg_test='1.0', isolated_clnt='False')


# distribute classes amongst client groups
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.45, rule_arg='0.2', rule_arg_test='1.0', isolated_clnt='False')

# doesn't distribute classes amongst client groups
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.0, rule_arg='0.3', rule_arg_test='1.0', isolated_clnt='False')
# data_obj = DatasetObject(dataset='CIFAR10', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.0, rule_arg='0.55', rule_arg_test='1.0', isolated_clnt='False')
# breakpoint()
###
model_name         = 'cifar10' # Model type
com_amount         = 100 # Number of communication rounds
save_period        = 10 # frequency of saving the model to disk; save_instances = com_amount/save_period
weight_decay       = 1e-3
batch_size         = 50
act_prob           = 1.0 # probability of client being active in a communication round
lr_decay_per_round = 0.9998
epoch              = 5 # client local training
learning_rate      = 0.1
print_per          = 1 # frequency of logging training updates 

# Model function
model_func = lambda : client_model(model_name)
# init_model = model_func()
# Initalise the model for all methods or load it from a saved initial model
init_model = model_func()
# print(init_model)
# exit()
if not os.path.exists('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    print("New directory!")
    os.mkdir('Output/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    
print('Output/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
# exit()
# Methods    
####


strat = 5
epoch = 5
seed = 60
dir_param_ =0.5
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
        # for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
#print("FedAvg EXPERIMENT")
#[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed)
#
#print("Prox EXPERIMENT")
#[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_Rational_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,mu = 1e-4,lr_decay_per_round=lr_decay_per_round,rand_seed = seed)
##
#print("IncFL EXPERIMENT")
#[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,rwpr, tst_perf_clnts] = train_rational_IncFL(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed)
#
#print("AGGFS EXPERIMENT")
#[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_AggFFs(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed)
   
ws_rounds = 4
print("FedAvg ws 4  EXPERIMENT")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,rand_seed = seed,ws_rounds = ws_rounds)

print("UCA EXPERIMENT")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,
 rwpr, tst_perf_clnts] = train_feduca(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, 
                                      print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round,rand_seed = seed,dir_param=dir_param_,num_strategies=strat,extra_thr = 0.0,val_ratio=0.0)