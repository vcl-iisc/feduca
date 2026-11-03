from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *
from utils_game import *
import time 
#from EPOSearch.mtc.epo_lp import *
from config import cfg 
import time
### Methods

def train_feduca(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0,dir_param=0.5,num_strategies=5,extra_thr = 0,val_ratio=1.0):
    # method_name = 'r2_m2_pretrained_rational_FedAvg_lr%s_%s_%s'%(cfg['lr'],cfg['epoch'],cfg['iid'])
    #method_name = 'convex_only_util_soft_reduce_random_uni_lottery_persuasion_FedAvg_solver_lambda_1.0_dir_'+str(dir_param)+'_seed_' + str(rand_seed)
    set_seed(rand_seed)
    method_name = 'feduca_accg_clnt_thr_actual_lg_acc'+ str(dir_param) + '_seed_' + str(rand_seed)
    method_name = method_name + '_thr_comm_' + str(extra_thr)+'_' + str(com_amount) + 'strat_' + str(num_strategies)
    #method_name = 'convex_only_util_soft_reduce_deterministic_lottery_persuasion_FedAvg_solver_lambda_1.0_dir_'+str(dir_param)+'_seed_' + str(rand_seed)
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y # client-wise data, for cifar10 with 10 clients, clnt_x shape (10,5000,3,32,32)
                                                     # and clnt_y shape (10,5000,1)
    rwpr = np.ones(com_amount).astype('float32') # round wise participation rate
    clnt_test_x = data_obj.clnt_tst_x; clnt_test_y = data_obj.clnt_tst_y

    cent_x = np.concatenate(clnt_x, axis=0) # for cifar10 with 10 clients, shape (50000,3,32,32)
    cent_y = np.concatenate(clnt_y, axis=0) # for cifar10 with 10 clients, shape (50000,1)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)]).astype('float32')
    weight_list = weight_list.reshape((n_clnt, 1)) # stores the number of data points per client to weigh them while aggregating
    acc_g_arr = np.zeros(n_clnt)

    P = np.ones((num_strategies,1)).astype('float32')/num_strategies 
    P_prev = np.ones((num_strategies,1)).astype('float32')/num_strategies 
    '''
    P is the probability assigned by server to different strategies.
    Here, #strategies = #clients.
    The #strategies can be different. Edit num_strategies to change.
    '''
    Q = np.ones((num_strategies, n_clnt)).astype('float32')/n_clnt # stores the weights assigned to each client for each strategy
    Q_prev = np.ones((num_strategies, n_clnt)).astype('float32')/n_clnt
    poly_deg = 2 # degree of the polynomial fitted to approximate the client utility
    clnt_util_params = np.zeros((n_clnt, poly_deg+1)).astype('float32') # to store the params of the polynomials fitted for each client
    clnt_util_params_prev = np.zeros((n_clnt, poly_deg+1)).astype('float32') # to store the params of the polynomials fitted for each client
    threshold = np.zeros((n_clnt, 1)).astype('float32') # to store the decision threshold for each client
    participation = np.zeros((n_clnt,)).astype('float32') # to track the individual client participation ratio

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
    cfg['output_dir'] = 'Output/%s/%s' %(data_obj.name, method_name)
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); # fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); # trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); # tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel_momentum = np.zeros((com_amount, 2))
    tst_perf_clnts = np.zeros((n_clnt,com_amount,2)) # to store the clients performance metrics on their text data
    uitl_fn_evol = np.zeros((com_amount, n_clnt, poly_deg+1)).astype('float32')
    threshold_evol = np.zeros((n_clnt, com_amount))
    participating_clients = np.zeros(n_clnt)
    n_par = len(get_mdl_params([model_func()])[0]) # to get an idea of n_par for the given task/data
    
    init_par_list=get_mdl_params([init_model], n_par)[0] # to create a parameter list of the init_model from saved parameters if available
    
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_params_list_rc=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # the params list server would use to update its model in case of rational clients
    clnt_models = list(range(n_clnt))
    clnt_models_rc = list(range(n_clnt))

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    avg_momentum_model = model_func().to(device)
    avg_momentum_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    alpha_param = [dir_param] * num_strategies
    strategy_vec = np.random.dirichlet(alpha_param,size=1).squeeze()
    print("strategy_vec:",strategy_vec)
    jgap = np.zeros((com_amount,n_clnt))

    empirical_utility =  np.zeros((com_amount,n_clnt,11))
    empirical_loss =  np.zeros((com_amount,n_clnt,11))
    fitted_utility =  np.zeros((com_amount,n_clnt,poly_deg+1))
    global_m_client_acc = np.zeros((com_amount,n_clnt))
    local_m_client_acc = np.zeros((com_amount,n_clnt))
    interpolated_acc = np.zeros(n_clnt)
    
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_sel_momentum = np.load('Output/%s/%s/%d_com_tst_perf_sel_momentum.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            # Client training 
            t1 = time.perf_counter()

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                
                tst_clnt_x = clnt_test_x[clnt]
                tst_clnt_y = clnt_test_y[clnt]
                
                
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters()))) # load global model params
                
                # rationality check
                if (i>0):
                    clnt_models_rc[clnt] = model_func().to(device)
                    clnt_models_rc[clnt] = set_client_from_params(clnt_models_rc[clnt], clnt_params_list[clnt])

                    loss_g, acc_g = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                    loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models_rc[clnt], data_obj.dataset)
                    tst_perf_clnts[clnt][i-1] = [loss_l, acc_l] # loss_l, acc_l for rationality check
                    global_m_client_acc[i,clnt] = acc_g
                    local_m_client_acc[i,clnt] = acc_l
                    print(f'zero shot loss of global model on client {clnt}: {loss_g}; accuracy: {acc_g}')
                    print(f'zero shot loss of local model on client {clnt}: {loss_l}; accuracy: {acc_l}')
                    weight_ass_prev = (P_prev.T@Q_prev)[0][clnt]
                    mdl_without_clnt = (global_mdl_params - (weight_ass_prev*clnt_params_list_rc[clnt]))/(1.0 - weight_ass_prev)

                    # carry out the sweep to get the utility curve
                    sweep = np.arange(0.0, 1.1, 0.1)
                    temp_data_fit = np.ones((len(sweep),)).astype(np.float32)
                    temp_data_fit_loss = np.ones((len(sweep),)).astype(np.float32)
                    weight_ass = (P.T@Q)[0][clnt]
                    exp_acc_on_fitted_poly = clnt_util_params_prev[clnt][0]*(weight_ass**2) + clnt_util_params_prev[clnt][1]*(weight_ass**1) + clnt_util_params_prev[clnt][2]
                    print("exp_acc_on_fitted_poly:", exp_acc_on_fitted_poly,"\t","threshold[clnt]:",threshold[clnt],"weight_ass:",weight_ass)
                    print("diff param:", clnt_util_params_prev[clnt]-clnt_util_params_prev[clnt])
                    util_is_lesser = (exp_acc_on_fitted_poly <= threshold[clnt])
                    if (util_is_lesser):    
                        print('loading the previously trained local model')
                        clnt_models[clnt].load_state_dict(copy.deepcopy(clnt_models_rc[clnt].state_dict()))
                        rwpr[i] = rwpr[i] - 1.0/n_clnt
                
                    clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                    clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                    #loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                    #print("i:",i,"\t","client:", clnt,"\t", "val acc:",acc_l)

                    for clnt_weight in sweep:
                        clnt_weight = round(clnt_weight, 1)
                        index = int(clnt_weight/0.1)
                        # evaluate the utility with sweeps on weights of the present model on the client
                        mdl_params_to_assess = (1.0 - clnt_weight)*mdl_without_clnt + clnt_weight*clnt_params_list[clnt]
                        mdl_to_assess = model_func().to(device)
                        mdl_to_assess = set_client_from_params(mdl_to_assess, mdl_params_to_assess)
                        temp_data_fit_loss[index], temp_data_fit[index] = get_acc_loss(tst_clnt_x, tst_clnt_y, mdl_to_assess, data_obj.dataset)
                        
                    if (i==(com_amount-1)):
                        interpolated_acc[clnt] = np.max(temp_data_fit)    

                    print("clnt",clnt, "\t", "temp_data_fit:", temp_data_fit)
                     
                    clnt_util_params[clnt] = fit_poly_clnt_util(temp_data_fit, sweep)
                    print("client:", clnt,'\t', "clnt_util_params:", clnt_util_params[clnt])
                    empirical_utility[i,clnt] = temp_data_fit.copy()
                    empirical_loss[i,clnt] = temp_data_fit_loss.copy()
                    fitted_utility[i,clnt] = clnt_util_params[clnt].copy()
                    clnt_util_params_prev[clnt] = clnt_util_params[clnt].copy()
                    threshold[clnt] = clnt_util_params[clnt][0]*(weight_ass**2) + clnt_util_params[clnt][1]*(weight_ass) + clnt_util_params[clnt][2]
                    a_tmp= clnt_util_params[clnt][0]*(weight_ass**2) + clnt_util_params[clnt][1]*(weight_ass) + clnt_util_params[clnt][2]                    
                    #threshold[clnt] = 0.4*(clnt_util_params[clnt][0]+ clnt_util_params[clnt][1] + clnt_util_params[clnt][2]) + 0.6*(acc_g)
                    #acc_g_arr[clnt] = acc_g
                    threshold[clnt] = np.maximum(0.01,threshold[clnt])
                    threshold[clnt] = np.minimum(0.99,threshold[clnt])
                    threshold_evol[clnt][i] = threshold[clnt]
               
                
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                
                if (i==0):
                    participation[clnt] = 1.0
                    participating_clients[clnt] = 1
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]
                elif(not util_is_lesser):    
                    participation[clnt] += 1.0
                    participating_clients[clnt] = 1
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]
                elif(util_is_lesser):
                    participating_clients[clnt] = 0

                if (i==(com_amount-1)):
                    for clnt in selected_clnts:
                        tst_clnt_x = clnt_test_x[clnt]
                        tst_clnt_y = clnt_test_y[clnt]

                        loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                        tst_perf_clnts[clnt][i] = [loss_l, acc_l]

                    print("interpolated_acc:",interpolated_acc)  

            uitl_fn_evol[i] = clnt_util_params.copy()
            
            print("client_participation_rate:", rwpr[i], "communication round:",i, "avg particpation:", np.mean(rwpr[:i]))
            
            if (i==0):
                avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list_rc[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
                avg_momentum_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters()))) 

                global_mdl_params = get_mdl_params([avg_model], n_par)[0]
            else:
                beta = participation/(i+1) 
                print(f'beta: {beta}')

                beta = 1/beta
                beta = beta/np.sum(beta)
                
                for b in range(len(beta)):
                    if beta[b] == 1.0:
                        beta[b] = beta[b] - 0.1

                print(f'1/beta: {beta}')
                              
                P, Q,jgap_vec  = adjust_Q_for_CC_feasibility(threshold, clnt_util_params, beta, num_strategies,participating_clients,strategy_vec,dir_param)
                #P, Q,jgap_vec  = adjust_Q_for_CC_feasibility_poly3(threshold, clnt_util_params, beta, num_strategies,participating_clients,strategy_vec,dir_param)
                is_feasible, P, Q = check_for_feasibility(threshold, clnt_util_params,P,Q)
                print("is_feasible:", is_feasible)
                
                if is_feasible == False:
                    print("no cllients meet threshold updated P and Q")
                
               
                jgap[i,:] = jgap_vec
                P = P.reshape(-1,1)

                P_prev = P.copy()
                Q_prev = Q.copy()
                print("P_prev:",P_prev)
                print("Q_prev:",Q_prev)

                
                weight_list[selected_clnts,0] = P.reshape(-1,1).T@Q
                #weight_list[selected_clnts,0] = np.sum(P_prev*Q_prev,axis = 0)
                print("np.sum(weight_list[selected_clnts]):",weight_list[selected_clnts])
                # aggregate based on the sampled strategy
                prev_model = get_mdl_params([avg_model], n_par)[0]
                avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list_rc[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
                global_mdl_params = get_mdl_params([avg_model], n_par)[0]
                avg_momentum_model = set_client_from_params(model_func(), np.sum([0.8*(prev_model),0.2*(global_mdl_params)], axis=0))
                
            ###
            # server performance on its validation set
            ########## compute the overall accuracy global model  #############
            loss_tst_server, acc_tst_server = get_acc_loss(data_obj.server_x[0], data_obj.server_y[0], avg_model, data_obj.dataset)
            #t2 = time.perf_counter()
            #print('t1-t2:', t2-t1)

            ########## compute the overall accuracy momentum model  #############     
            loss_tst_server_momentum, acc_tst_server_momentum = get_acc_loss(data_obj.server_x[0], data_obj.server_y[0], avg_momentum_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst_server, acc_tst_server]
            print("**** Communication sel %3d, Server Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst_server, loss_tst_server))
            ###
            # server average performance on all of client data - server model generalization performance
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Server Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            tst_perf_sel_momentum[i] = [loss_tst_server_momentum, acc_tst_server_momentum]
            print("**** Communication sel %3d, Server Momentum Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst_server_momentum, loss_tst_server_momentum))
            ###

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                torch.save(avg_momentum_model.state_dict(), 'Output/%s/%s/%d_com_sel_momentum.pt' %(data_obj.name, method_name, (i+1)))  

                # torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                #np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, (i+1)), tst_perf_clnts[:, :i+1, :])
                np.save('Output/%s/%s/%d_com_tst_perf_sel_momentum.npy' % (data_obj.name, method_name, (i+1)), tst_perf_sel_momentum[:i+1])
                np.save('Output/%s/%s/%d_com_jgap.npy' % (data_obj.name, method_name, (i+1)), jgap[:i+1])

                np.save('Output/%s/%s/%d_empirical_loss.npy' % (data_obj.name, method_name, (i+1)), empirical_loss)
                np.save('Output/%s/%s/%d_empirical_utility.npy' % (data_obj.name, method_name, (i+1)), empirical_utility)
                np.save('Output/%s/%s/%d_fitted_utility.npy' % (data_obj.name, method_name, (i+1)), fitted_utility)
                np.save('Output/%s/%s/%d_global_m_client_acc.npy' % (data_obj.name, method_name, (i+1)), global_m_client_acc)
                np.save('Output/%s/%s/%d_local_m_client_acc.npy' % (data_obj.name, method_name, (i+1)), local_m_client_acc)
                

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel_momentum.npy' % (data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                # fed_mdls_all[i//save_period] = all_model
    if not os.path.exists('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, (i+1)), rwpr)
    if not os.path.exists('Output/%s/%s/%d_util_fn_evol.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_util_fn_evol.npy' %(data_obj.name, method_name, (i+1)), uitl_fn_evol)
    if not os.path.exists('Output/%s/%s/%d_threshold_evol.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_threshold_evol.npy' %(data_obj.name, method_name, (i+1)), threshold_evol)

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, rwpr, tst_perf_clnts #, fed_mdls_all, trn_perf_all, tst_perf_all


def train_rational_FedAvg(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0,ws_rounds=-1):
    # method_name = 'r2_m2_pretrained_rational_FedAvg_lr%s_%s_%s'%(cfg['lr'],cfg['epoch'],cfg['iid'])
    set_seed(rand_seed)
    method_name = 'nips26_RFedAvg_iid%s_n_clnts%s_bl%s_%s'%(cfg['iid'],cfg['n_clnts'], cfg['bl'], rand_seed)
    if ws_rounds > 0:
        method_name = "wsn_" + str(ws_rounds) + '_' + method_name
    print(method_name)
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y # client-wise data, for cifar10 with 10 clients, clnt_x shape (10,5000,3,32,32)
                                                     # and clnt_y shape (10,5000,1)
    rwpr = np.ones(com_amount).astype('float32') # round wise participation rate
    clnt_test_x = data_obj.clnt_tst_x; clnt_test_y = data_obj.clnt_tst_y

    cent_x = np.concatenate(clnt_x, axis=0) # for cifar10 with 10 clients, shape (50000,3,32,32)
    cent_y = np.concatenate(clnt_y, axis=0) # for cifar10 with 10 clients, shape (50000,1)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)]) 
    weight_list = weight_list.reshape((n_clnt, 1)) # stores the number of data points per client to weigh them while aggregating

    acc_g_arr = np.zeros(n_clnt)

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
    cfg['output_dir'] = 'Output/%s/%s' %(data_obj.name, method_name)
    print(cfg['output_dir'])
    # exit()
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); # fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); # trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); # tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_clnts = np.zeros((n_clnt,com_amount,2)) # to store the clients performance metrics on their text data
    
    n_par = len(get_mdl_params([model_func()])[0]) # to get an idea of n_par for the given task/data
    
    init_par_list=get_mdl_params([init_model], n_par)[0] # to create a parameter list of the init_model from saved parameters if available
    
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_params_list_rc=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # the params list server would use to update its model in case of rational clients
    clnt_models = list(range(n_clnt))
    clnt_models_rc = list(range(n_clnt))

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    interpolate_client_acc = np.zeros(n_clnt)

    # all_model = model_func().to(device)
    # all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            # fed_model = model_func()
            # fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            # fed_model.eval()
            # fed_model = fed_model.to(device)
            # fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnts = np.load('Output/%s/%s/%d_com_tst_perf_clnts.npy' %(data_obj.name, method_name, com_amount))
        rwpr = np.load('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            # Client training 
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                tst_clnt_x = clnt_test_x[clnt]
                tst_clnt_y = clnt_test_y[clnt]

                # print(tst_clnt_x.shape)
                # breakpoint()
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters()))) # load global model params
                
                # rationality check
                if (i>0):
                    # print(tst_clnt_x)
                    # exit()
                    clnt_models_rc[clnt] = model_func().to(device)
                    clnt_models_rc[clnt] = set_client_from_params(clnt_models_rc[clnt], clnt_params_list[clnt])

                    loss_g, acc_g = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                    loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models_rc[clnt], data_obj.dataset)
                    tst_perf_clnts[clnt][i-1] = [loss_l, acc_l] # loss_l, acc_l for rationality check

                    print(f'zero shot loss of global model on client {clnt}: {loss_g}; accuracy: {acc_g}')
                    print(f'zero shot loss of local model on client {clnt}: {loss_l}; accuracy: {acc_l}')

                    delta_rc = loss_l - loss_g
                    #delta_rc = acc_g - (0.55*acc_g_arr[clnt] + 0.45*acc_l)
                    acc_g_arr[clnt] = acc_g
                    if (delta_rc < 0 and i > ws_rounds):
                        print('loading the previously trained local model')
                        clnt_models[clnt].load_state_dict(copy.deepcopy(clnt_models_rc[clnt].state_dict()))
                        rwpr[i] = rwpr[i] - 1.0/n_clnt
                        
                    # breakpoint()
                # -----------------
                
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                #t1 = time.time()
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                #t2 = time.time()
                #print("fedavg t2-t1:", t2-t1)
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # clients choosing to participate or not based on their rationality
                
                if (i==0 ):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]
                elif (delta_rc>0 or i <= ws_rounds):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]

                if (i==(com_amount-1)):
                    for clnt in selected_clnts:
                        tst_clnt_x = clnt_test_x[clnt]
                        tst_clnt_y = clnt_test_y[clnt]

                        loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                        tst_perf_clnts[clnt][i] = [loss_l, acc_l]
                        interpolate_client_acc[clnt] = interpolate_test(tst_clnt_x, tst_clnt_y, model_func,clnt_models[clnt],avg_model,  data_obj.dataset, n_par)
                        interpolate_client_acc[clnt] = np.maximum(interpolate_client_acc[clnt],acc_l)
                    print("best_client_acc:",interpolate_client_acc)    

            # Client training ends
            # Server computation follows
            print(f'client_participation_rate in {i}th communication round: {rwpr[i]}')
            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list_rc[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            # all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            # server performance on its validation set
            # loss_tst_server, acc_tst_server = get_acc_loss(data_obj.server_x[0], data_obj.server_y[0], avg_model, data_obj.dataset)
            print(data_obj.server_x.squeeze().shape)
            # exit()
            loss_tst_server, acc_tst_server = get_acc_loss(data_obj.server_x.squeeze(), data_obj.server_y.squeeze(), avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst_server, acc_tst_server]
            print("**** Communication sel %3d, Server Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst_server, loss_tst_server))
            ###
            # server average performance on all of client data - server model generalization performance
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Server Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))


            ###
            # loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            # tst_perf_all[i] = [loss_tst, acc_tst]
            # print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            # ###
            # loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            # trn_perf_all[i] = [loss_tst, acc_tst]
            # print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                # torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, (i+1)), tst_perf_clnts[:, :i+1, :])

                # np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                # np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                # fed_mdls_all[i//save_period] = all_model
    if not os.path.exists('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, (i+1)), rwpr)

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, rwpr, tst_perf_clnts #, fed_mdls_all, trn_perf_all, tst_perf_all



def train_rational_AggFFs(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0,ws_rounds=-1):
    method_name = 'nips26_rational_game_ONS_strategy'
    method_name =  method_name + 'seed_'+str(rand_seed)
    
    if ws_rounds > 0:
        method_name = "ws_" + str(ws_rounds) + '_' + method_name + 'seed_'+str(rand_seed)
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y # client-wise data, for cifar10 with 10 clients, clnt_x shape (10,5000,3,32,32)
                                                     # and clnt_y shape (10,5000,1)
    rwpr = np.ones(com_amount).astype('float32') # round wise participation rate
    clnt_test_x = data_obj.clnt_tst_x; clnt_test_y = data_obj.clnt_tst_y

    cent_x = np.concatenate(clnt_x, axis=0) # for cifar10 with 10 clients, shape (50000,3,32,32)
    cent_y = np.concatenate(clnt_y, axis=0) # for cifar10 with 10 clients, shape (50000,1)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)]) 
    weight_list = weight_list.reshape((n_clnt, 1)) # stores the number of data points per client to weigh them while aggregating
    modulated_weight_list = np.asarray([1.0] * n_clnt).astype('float32') # stores the modulated weight by server when it plays the game
    modulated_weight_list = modulated_weight_list.reshape((n_clnt, 1))

    weight_list_req = np.asarray([1.0] * n_clnt).astype('float32') # the weight_list requested by clients
    weight_list_req = weight_list_req.reshape((n_clnt, 1))
    prev_weight_list = np.asarray([1.0] * n_clnt).astype('float32')/n_clnt # to store the weights the server used in the previous round
    prev_weight_list = prev_weight_list.reshape((n_clnt, 1))
    
    loss_gminusone = np.asarray([1.0] * n_clnt).astype('float32') # to store the previous round global model zero shot accuracy on clients val data
    loss_gminusone = loss_gminusone.reshape((n_clnt,1))
    positive_cor_prob = np.zeros((n_clnt, n_clnt), dtype=np.float32)
    negative_cor_prob = np.zeros((n_clnt, n_clnt), dtype=np.float32)

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); # fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); # trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); # tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_clnts = np.zeros((n_clnt,com_amount,2)) # to store the clients performance metrics on their text data
    clnt_participation = np.ones((n_clnt,com_amount), dtype=np.float32) # to store each client participation statistics
    weight_evolution = np.zeros((com_amount,n_clnt), dtype=np.float32) # to store the weights assigned to clients over the training
    n_par = len(get_mdl_params([model_func()])[0]) # to get an idea of n_par for the given task/data  
    init_par_list=get_mdl_params([init_model], n_par)[0] # to create a parameter list of the init_model from saved parameters if available
    
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_params_list_rc=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # the params list server would use to update its model in case of rational clients
    clnt_models = list(range(n_clnt))
    clnt_models_rc = list(range(n_clnt))
   
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    acc_g_arr = np.zeros(n_clnt)
    # borrowing params from official implementation: https://github.com/vaseline555/AAggFF/blob/main/src/server/aaggffadagradserver.py
    lipschitz = 1.0 / n_clnt
    alpha = 4.0 * n_clnt * lipschitz
    beta = 1.0 / (4.0 * lipschitz)

    ons_weights = np.zeros((n_clnt, n_clnt), dtype=np.float32) # to store the regret-minimizing strategy weights of clients
    grad_history = np.zeros((n_clnt, n_clnt), dtype=np.float32) # to accumulate the gradient updates for the ONS update
    hess_history = alpha * np.eye(n_clnt, dtype=np.float32)[np.newaxis, :, :].repeat(n_clnt, axis=0) # to accumulate the hessian updates for the ONS update
    interpolate_client_acc = np.zeros(n_clnt)
    # all_model = model_func().to(device)
    # all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            # fed_model = model_func()
            # fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            # fed_model.eval()
            # fed_model = fed_model.to(device)
            # fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        #trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        #tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnts = np.load('Output/%s/%s/%d_com_tst_perf_clnts.npy' %(data_obj.name, method_name, com_amount))

        rwpr = np.load('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount))
        clnt_participation = np.load('Output/%s/%s/%d_client_participation.npy' %(data_obj.name, method_name, com_amount))
        weight_evolution = np.load('Output/%s/%s/%d_weight_evolution.npy' %(data_obj.name, method_name, com_amount))

        positive_cor_prob = np.load('Output/%s/%s/%d_com_positive_cor_prob.npy' %(data_obj.name, method_name, com_amount))
        negative_cor_prob = np.load('Output/%s/%s/%d_com_negative_cor_prob.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            # Client training 
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                tst_clnt_x = clnt_test_x[clnt]
                tst_clnt_y = clnt_test_y[clnt]
                # breakpoint()
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters()))) # load global model params
                
                # rationality check
                if (i>0):
                    if(i==1):
                        loss_gminusone[clnt] = 0.0

                    clnt_models_rc[clnt] = model_func().to(device)
                    clnt_models_rc[clnt] = set_client_from_params(clnt_models_rc[clnt], clnt_params_list[clnt])

                    loss_g, acc_g = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                    loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models_rc[clnt], data_obj.dataset)
                    tst_perf_clnts[clnt][i-1] = [loss_l, acc_l]

                    print(f'zero shot loss of global model on client {clnt}: {loss_g}; accuracy: {acc_g}')
                    print(f'zero shot loss of local model on client {clnt}: {loss_l}; accuracy: {acc_l}')

                    #G2: calculation of knowledge incentive
                    # breakpoint()
                    #p_k, positive_cor_prob, negative_cor_prob = calculate_clnt_knowledge_inc(modulated_weight_list[selected_clnts], prev_weight_list[selected_clnts],
                    #                                                clnt, loss_g, loss_gminusone[clnt], positive_cor_prob, negative_cor_prob)
                    loss_gminusone[clnt] = loss_g
                    p_k = 0
                    delta = loss_l - loss_g
                    delta_rc = delta + p_k # added client's knowledge incentive to its rationality
                    #delta_rc = acc_g - (0.6*acc_g_arr[clnt]+0.4*acc_l)
                    #acc_g_arr[clnt] = acc_g
            
                    if (delta_rc < 0 and i  > ws_rounds):
                        print('loading the previously trained local model')
                        clnt_models[clnt].load_state_dict(copy.deepcopy(clnt_models_rc[clnt].state_dict()))
                        rwpr[i] = rwpr[i] - 1.0/n_clnt
                        clnt_participation[clnt][i] = 0.0
                        
                    # breakpoint()
                # -----------------
                
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                
                t1 = time.time()    
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                t2 = time.time()
                
                print("aaggff-s:", t2-t1)
                
                if (i==0):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]
                elif (delta_rc>0 or i <= ws_rounds):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]

                if (i==(com_amount-1)):
                    for clnt in selected_clnts:
                        tst_clnt_x = clnt_test_x[clnt]
                        tst_clnt_y = clnt_test_y[clnt]

                        loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                        tst_perf_clnts[clnt][i] = [loss_l, acc_l]

                        interpolate_client_acc[clnt] = interpolate_test(tst_clnt_x, tst_clnt_y, model_func,clnt_models[clnt],avg_model,  data_obj.dataset, n_par)
                        interpolate_client_acc[clnt] = np.maximum(interpolate_client_acc[clnt],acc_l)
                    print("best_client_acc:",interpolate_client_acc)    

                        

            if (i>0):
                print('updating the previous weight list')
                prev_weight_list[selected_clnts] = modulated_weight_list[selected_clnts]

                # G4: ONS weight calculation
                ons_weights, grad_history, hess_history = get_ONS_weights(modulated_weight_list, tst_perf_clnts, 
                                                i-1, n_clnt, positive_cor_prob, negative_cor_prob, 
                                                grad_history, hess_history, alpha = alpha, beta = beta)
            # Client training ends
            # Server computation follows
            print(f'client_participation_rate in {i}th communication round: {rwpr[i]}')
            # Scale with weights
            # G1: Server modulates the weights as per its data
            # breakpoint()
            if (i==0):
                weight_list_req[selected_clnts] = weight_list[selected_clnts]/np.sum(weight_list[selected_clnts])
            else:
                weight_list_req = solve_ONS_weight_request(ons_weights)
            #modulated_weight_list[selected_clnts] = server_modulated_weights(weight_list_req[selected_clnts], 
            #                                                                 data_obj.server_x[0], data_obj.server_y[0], 
            #                                                                 clnt_params_list_rc[selected_clnts],
            #                                                                 model_func, data_obj)
              
            modulated_weight_list = weight_list_req.reshape(-1,1)
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list_rc[selected_clnts]*modulated_weight_list[selected_clnts], axis = 0))
            weight_evolution[i][selected_clnts] = modulated_weight_list[selected_clnts].squeeze(1)
            # all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            # server performance on its validation set
            loss_tst_server, acc_tst_server = get_acc_loss(data_obj.server_x[0], data_obj.server_y[0], avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst_server, acc_tst_server]
            print("**** Communication sel %3d, Server Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst_server, loss_tst_server))
            ###
            # server average performance on all of client data - server model generalization performance
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Server Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))


            ###
            # loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, all_model, data_obj.dataset)
            # tst_perf_all[i] = [loss_tst, acc_tst]
            # print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))
            # ###
            # loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset)
            # trn_perf_all[i] = [loss_tst, acc_tst]
            # print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                # torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, (i+1)), tst_perf_clnts[:, :i+1, :])
                np.save('Output/%s/%s/%d_com_positive_cor_prob.npy' % (data_obj.name, method_name, (i+1)), positive_cor_prob)
                np.save('Output/%s/%s/%d_com_negative_cor_prob.npy' % (data_obj.name, method_name, (i+1)), negative_cor_prob)
                

                # np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                # np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_positive_cor_prob.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_negative_cor_prob.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                # fed_mdls_all[i//save_period] = all_model

    if not os.path.exists('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, (i+1)), rwpr)
    
    if not os.path.exists('Output/%s/%s/%d_client_participation.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_client_participation.npy' %(data_obj.name, method_name, (i+1)), clnt_participation)

    if not os.path.exists('Output/%s/%s/%d_weight_evolution.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_weight_evolution.npy' %(data_obj.name, method_name, (i+1)), weight_evolution)

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, rwpr, tst_perf_clnts #, fed_mdls_all, trn_perf_all, tst_perf_all



def train_rational_IncFL(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, lr_decay_per_round, rand_seed=0,ws_rounds=-1):
    # method_name = 'corrected_rational_IncFL_c+_0.3_alpha_0.6'
    set_seed(rand_seed)
    method_name = 'nips26_2_RFedINCFL_iid%s_n_clnts%s_bl%s_%s'%(cfg['iid'],cfg['n_clnts'], cfg['bl'], rand_seed)
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y # client-wise data, for cifar10 with 10 clients, clnt_x shape (10,5000,3,32,32)
                                                     # and clnt_y shape (10,5000,1)
    rwpr = np.ones(com_amount).astype('float32') # round wise participation rate
    clnt_test_x = data_obj.clnt_tst_x; clnt_test_y = data_obj.clnt_tst_y

    cent_x = np.concatenate(clnt_x, axis=0) # for cifar10 with 10 clients, shape (50000,3,32,32)
    cent_y = np.concatenate(clnt_y, axis=0) # for cifar10 with 10 clients, shape (50000,1)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)]).astype('float32') 
    weight_list = weight_list.reshape((n_clnt, 1)) # stores the number of data points per client to weigh them while aggregating

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); # fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); # trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); # tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_clnts = np.zeros((n_clnt,com_amount,2)) # to store the clients performance metrics on their text data
    
    n_par = len(get_mdl_params([model_func()])[0]) # to get an idea of n_par for the given task/data
     
    init_par_list=get_mdl_params([init_model], n_par)[0] # to create a parameter list of the init_model from saved parameters if available
    
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_params_list_rc=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # the params list server would use to update its model in case of rational clients
    clnt_models = list(range(n_clnt))
    clnt_models_rc = list(range(n_clnt))
    acc_g_arr = np.zeros(n_clnt)
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    interpolate_client_acc = np.zeros(n_clnt)
    # all_model = model_func().to(device)
    # all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
            # fed_model = model_func()
            # fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            # fed_model.eval()
            # fed_model = fed_model.to(device)
            # fed_mdls_all[j] = fed_model
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnts = np.load('Output/%s/%s/%d_com_tst_perf_clnts.npy' %(data_obj.name, method_name, com_amount))
        rwpr = np.load('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount))
        
    else:
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))

            # Client training 
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                tst_clnt_x = clnt_test_x[clnt]
                tst_clnt_y = clnt_test_y[clnt]
                # breakpoint()
                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters()))) # load global model params
                
                # rationality check
                if (i>0):
                    clnt_models_rc[clnt] = model_func().to(device)
                    clnt_models_rc[clnt] = set_client_from_params(clnt_models_rc[clnt], clnt_params_list[clnt])

                    loss_g, acc_g = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                    loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models_rc[clnt], data_obj.dataset)
                    tst_perf_clnts[clnt][i-1] = [loss_l, acc_l]

                    print(f'zero shot loss of global model on client {clnt}: {loss_g}; accuracy: {acc_g}')
                    print(f'zero shot loss of local model on client {clnt}: {loss_l}; accuracy: {acc_l}')

                    delta_rc = loss_l - loss_g
                    #delta_rc = acc_g - (0.4*acc_l+0.6*acc_g_arr[clnt])
                    acc_g_arr[clnt] = acc_g
                    if (delta_rc < 0 and i > ws_rounds):
                        print('loading the previously trained local model')
                        clnt_models[clnt].load_state_dict(copy.deepcopy(clnt_models_rc[clnt].state_dict()))
                        rwpr[i] = rwpr[i] - 0.1 # 1.0/num_clients
                        
                    # breakpoint()
                # -----------------
                
                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                t1 = time.time()
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                t2 = time.time()
                
                print("t2-t1 incfl:", t2-t1)
                # clients choosing to participate or not based on their rationality
                
                if (i==0 ):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]
                elif (delta_rc>0 or i <= ws_rounds):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]

                if (i==(com_amount-1)):
                    for clnt in selected_clnts:
                        tst_clnt_x = clnt_test_x[clnt]
                        tst_clnt_y = clnt_test_y[clnt]

                        loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                        tst_perf_clnts[clnt][i] = [loss_l, acc_l]
                        interpolate_client_acc[clnt] = interpolate_test(tst_clnt_x, tst_clnt_y, model_func,clnt_models[clnt],avg_model,  data_obj.dataset, n_par)
                        interpolate_client_acc[clnt] = np.maximum(interpolate_client_acc[clnt],acc_l)
                    print("best_client_acc:",interpolate_client_acc)        

                # IncFL update weights
                
                if (i>0):  
                    sigma = 1.0 / (1.0 + np.exp(-delta_rc))
                    weight_list[clnt] = sigma*(1-sigma)

            # Client training ends
            # Server computation follows
            print(f'client_participation_rate in {i}th communication round: {rwpr[i]}')
            # Scale with weights
            
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list_rc[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            # all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            # server performance on its validation set
            loss_tst_server, acc_tst_server = get_acc_loss(data_obj.server_x[0], data_obj.server_y[0], avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst_server, acc_tst_server]
            print("**** Communication sel %3d, Server Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst_server, loss_tst_server))
            ###
            # server average performance on all of client data - server model generalization performance
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Server Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))


            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                # torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, (i+1)), tst_perf_clnts[:, :i+1, :])

                # np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                # np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                # fed_mdls_all[i//save_period] = all_model
    if not os.path.exists('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount)):
        np.save('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, (i+1)), rwpr)

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, rwpr, tst_perf_clnts #, fed_mdls_all, trn_perf_all, tst_perf_all





def train_Rational_FedProx(data_obj, act_prob ,learning_rate, batch_size, epoch, com_amount, print_per, weight_decay, model_func, init_model, save_period, mu, lr_decay_per_round, rand_seed=0,ws_rounds=-1):

    set_seed(cfg['seed'])
    method_name = 'nips26_nr_obj_2_RFedProx_iid%s_n_clnts%s_bl%s_%s'%(cfg['iid'],cfg['n_clnts'], cfg['bl'], cfg['seed'])
    method_name = method_name + 'seed_' + str(rand_seed) 
    if ws_rounds > 0:
        method_name = "ws_" + str(ws_rounds) + '_' + method_name
    print(method_name)
    print('Output/%s/%s/' %(data_obj.name, method_name))
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y # client-wise data, for cifar10 with 10 clients, clnt_x shape (10,5000,3,32,32)
                                                     # and clnt_y shape (10,5000,1)
    rwpr = np.ones(com_amount).astype('float32') # round wise participation rate
    clnt_test_x = data_obj.clnt_tst_x; clnt_test_y = data_obj.clnt_tst_y

    cent_x = np.concatenate(clnt_x, axis=0) # for cifar10 with 10 clients, shape (50000,3,32,32)
    cent_y = np.concatenate(clnt_y, axis=0) # for cifar10 with 10 clients, shape (50000,1)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)]) 
    weight_list = weight_list.reshape((n_clnt, 1)) # stores the number of data points per client to weigh them while aggregating

    interpolate_client_acc = np.zeros(n_clnt)
    acc_g_arr = np.zeros(n_clnt)

    if not os.path.exists('Output/%s/%s' %(data_obj.name, method_name)):
        os.mkdir('Output/%s/%s' %(data_obj.name, method_name))
    cfg['output_dir'] = 'Output/%s/%s' %(data_obj.name, method_name)
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); # fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 2)); # trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2)); # tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_clnts = np.zeros((n_clnt,com_amount,2)) # to store the clients performance metrics on their text data
    
    n_par = len(get_mdl_params([model_func()])[0]) # to get an idea of n_par for the given task/data
     
    init_par_list=get_mdl_params([init_model], n_par)[0] # to create a parameter list of the init_model from saved parameters if available
    
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    clnt_params_list_rc=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # the params list server would use to update its model in case of rational clients
    clnt_models = list(range(n_clnt))
    clnt_models_rc = list(range(n_clnt))

    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    # all_model = model_func().to(device)
    # all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    if os.path.exists('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount)):
        # Load performances and models...
        for j in range(n_save_instances):
            fed_model = model_func()
            fed_model.load_state_dict(torch.load('Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (j+1)*save_period)))
            fed_model.eval()
            fed_model = fed_model.to(device)
            fed_mdls_sel[j] = fed_model
            
        
        trn_perf_sel = np.load('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # trn_perf_all = np.load('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, com_amount))

        tst_perf_sel = np.load('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, com_amount))
        # tst_perf_all = np.load('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, com_amount))
        tst_perf_clnts = np.load('Output/%s/%s/%d_com_tst_perf_clnts.npy' %(data_obj.name, method_name, com_amount))
        clnt_params_list = np.load('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, com_amount))
        rwpr = np.load('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount))
    else:
    
        for i in range(com_amount):

            inc_seed = 0
            while(True):
                # Fix randomness in client selection
                np.random.seed(i + rand_seed + inc_seed)
                act_list    = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                if len(selected_clnts) != 0:
                    break
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            avg_model_param = get_mdl_params([avg_model], n_par)[0]
            avg_model_param_tensor = torch.tensor(avg_model_param, dtype=torch.float32, device=device)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                tst_clnt_x = clnt_test_x[clnt]
                tst_clnt_y = clnt_test_y[clnt]

                clnt_models[clnt] = model_func().to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                # rationality check
                if (i>0):
                    # print(tst_clnt_x)
                    # exit()
                    clnt_models_rc[clnt] = model_func().to(device)
                    clnt_models_rc[clnt] = set_client_from_params(clnt_models_rc[clnt], clnt_params_list[clnt])

                    loss_g, acc_g = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                    loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models_rc[clnt], data_obj.dataset)
                    tst_perf_clnts[clnt][i-1] = [loss_g, acc_g] # loss_l, acc_l for rationality check

                    print(f'zero shot loss of global model on client {clnt}: {loss_g}; accuracy: {acc_g}')
                    print(f'zero shot loss of local model on client {clnt}: {loss_l}; accuracy: {acc_l}')

                    delta_rc = loss_l - loss_g
                    #delta_rc = acc_g - (0.4*acc_l + 0.6*acc_g_arr[clnt])
                    #acc_g_arr[clnt]= acc_g
                    if (delta_rc < 0 and i > ws_rounds):
                        print('loading the previously trained local model')
                        clnt_models[clnt].load_state_dict(copy.deepcopy(clnt_models_rc[clnt].state_dict()))
                        rwpr[i] = rwpr[i] - 0.1 # 1.0/num_clients

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                
                t1 = time.time()    
                clnt_models[clnt] = train_fedprox_mdl(clnt_models[clnt], avg_model_param_tensor, mu, trn_x, trn_y, learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per, weight_decay, data_obj.dataset)
                t2 = time.time()
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par)[0]
                # clients choosing to participate or not based on their rationality
                print("fedprox:", t2-t1)
                if (i==0):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]
                elif (delta_rc>0 or i <=ws_rounds):
                    clnt_params_list_rc[clnt] = clnt_params_list[clnt]
                
                if (i==(com_amount-1)):
                    for clnt in selected_clnts:
                        tst_clnt_x = clnt_test_x[clnt]
                        tst_clnt_y = clnt_test_y[clnt]

                        loss_l, acc_l = get_acc_loss(tst_clnt_x, tst_clnt_y, clnt_models[clnt], data_obj.dataset)
                        tst_perf_clnts[clnt][i] = [loss_l, acc_l]

                        interpolate_client_acc[clnt] = interpolate_test(tst_clnt_x, tst_clnt_y, model_func,clnt_models[clnt],avg_model,  data_obj.dataset, n_par)
                        interpolate_client_acc[clnt] = np.maximum(interpolate_client_acc[clnt],acc_l)
                    print("best_client_acc:",interpolate_client_acc)    


            # Scale with weights
            avg_model = set_client_from_params(model_func(), np.sum(clnt_params_list[selected_clnts]*weight_list[selected_clnts]/np.sum(weight_list[selected_clnts]), axis = 0))
            # all_model = set_client_from_params(model_func(), np.sum(clnt_params_list*weight_list/np.sum(weight_list), axis = 0))

            ###
            # loss_tst, acc_tst = get_acc_loss(data_obj.clnt_tst_x, data_obj.clnt_tst_y, avg_model, data_obj.dataset)
            # tst_perf_sel[i] = [loss_tst, acc_tst]
            loss_tst_server, acc_tst_server = get_acc_loss(data_obj.server_x.squeeze(), data_obj.server_y.squeeze(), avg_model, data_obj.dataset)
            tst_perf_sel[i] = [loss_tst_server, acc_tst_server]
            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst_server, loss_tst_server))
            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f" %(i+1, acc_tst, loss_tst))

            if ((i+1) % save_period == 0):
                torch.save(avg_model.state_dict(), 'Output/%s/%s/%d_com_sel.pt' %(data_obj.name, method_name, (i+1))) 
                # torch.save(all_model.state_dict(), 'Output/%s/%s/%d_com_all.pt' %(data_obj.name, method_name, (i+1)))
                np.save('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, (i+1)), clnt_params_list)

                np.save('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, (i+1)), trn_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, (i+1)), tst_perf_sel[:i+1])
                np.save('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, (i+1)), tst_perf_clnts[:, :i+1, :])

                # np.save('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, (i+1)), trn_perf_all[:i+1])
                # np.save('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, (i+1)), tst_perf_all[:i+1])

                if (i+1) > save_period:
                    if os.path.exists('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Output/%s/%s/%d_com_trn_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_sel.npy' %(data_obj.name, method_name, i+1-save_period))
                        os.remove('Output/%s/%s/%d_com_tst_perf_clnts.npy' % (data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_trn_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))
                        # os.remove('Output/%s/%s/%d_com_tst_perf_all.npy' %(data_obj.name, method_name, i+1-save_period))

                        os.remove('Output/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, method_name, i+1-save_period))

            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                # fed_mdls_all[i//save_period] = all_model
        if not os.path.exists('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, com_amount)):
            np.save('Output/%s/%s/%d_client_rwpr.npy' %(data_obj.name, method_name, (i+1)), rwpr)
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, rwpr, tst_perf_clnts #, fed_mdls_all, trn_perf_all, tst_perf_all