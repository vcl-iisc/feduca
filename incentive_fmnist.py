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


al = cfg['al']
bh = cfg['bh']


strat = 5
n_client = 10
data_obj = DatasetObject(dataset='fmnist', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.0, rule_arg='1.0', rule_arg_test='1.0', isolated_clnt='False')
# data_obj = DatasetObject(dataset='fmnist', n_client=n_client, rule='Dirichlet', unbalanced_sgm=0.0, rule_arg='0.55', rule_arg_test='1.0', isolated_clnt='False')
# breakpoint()

###
model_name         = 'fmnist' # Model type
com_amount         = 100 # Number of communication rounds
save_period        = 20 # frequency of saving the model to disk; save_instances = com_amount/save_period
weight_decay       = 1e-3
batch_size         = 50
act_prob           = 1 # probability of client being active in a communication round
lr_decay_per_round = 0.9998
epoch              = 5 # client local training
learning_rate      = 0.1
print_per          = 1 # frequency of logging training updates 
#seed = cfg['seed']

seed = 10
# breakpoint()
# Model function
model_func = lambda : client_model(model_name)
# init_model = model_func()
# Initalise the model for all methods or load it from a saved initial model
init_model = model_func()

ws_rounds = -1
print("Rational FedAvg")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,ws_rounds = ws_rounds)

print("Rational FedProx")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_Rational_FedProx(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,mu = 1e-4,lr_decay_per_round=lr_decay_per_round)
#
print("Rational IncFL")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,rwpr, tst_perf_clnts] = train_rational_IncFL(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round)

print("Rational aggFL")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_AggFFs(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round)

ws_rounds = 4
print("Rational fedavg warmstart")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg, rwpr, tst_perf_clnts] = train_rational_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,model_func=model_func, init_model=init_model, save_period=save_period,lr_decay_per_round=lr_decay_per_round,ws_rounds = ws_rounds)

print("Rational FedUCA")
[fed_mdls_sel_rational_FedAvg, trn_perf_sel_rational_FedAvg, tst_perf_sel_rational_FedAvg,
 rwpr, tst_perf_clnts] = train_feduca(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size,
                                     epoch=epoch, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                                     model_func=model_func, init_model=init_model, save_period=save_period,
                                     lr_decay_per_round=lr_decay_per_round,rand_seed = seed,dir_param=0.5,num_strategies=strat)
