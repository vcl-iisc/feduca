from utils_libs import *
import urllib.request
import zipfile
import shutil

class DatasetObject:
    def __init__(self, dataset, n_client, rule, unbalanced_sgm=0, rule_arg='', rule_arg_test='', isolated_clnt=False):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.rule_arg_test = rule_arg_test
        rule_arg_str  = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        rule_arg_test_str  = rule_arg_test if isinstance(rule_arg_test, str) else '%.3f' % rule_arg_test
        self.name = "%s_%d_%s_%s_%s" %(self.dataset, self.n_client, self.rule, rule_arg_str, rule_arg_test_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.isolated_clnt = isolated_clnt
        self.data_path = 'Data'
        self.set_data()
        
    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%s/%s' %(self.data_path, self.name)):
            # Get Raw data                
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='%s/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='%s/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'fmnist':
                print('loading fmnist')
                # Standard grayscale transform: tensor + channel‑wise normalization
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.2860,), std=(0.3530,))
                ])

                trnset = torchvision.datasets.FashionMNIST(root=f'{self.data_path}/Raw',
                                                        train=True, download=True, transform=transform)
                tstset = torchvision.datasets.FashionMNIST(root=f'{self.data_path}/Raw',
                                                        train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(
                    trnset, batch_size=60000, shuffle=False, num_workers=1
                )
                tst_load = torch.utils.data.DataLoader(
                    tstset, batch_size=10000, shuffle=False, num_workers=1
                )

                self.channels = 1; self.width    = 28; self.height   = 28; self.n_cls    = 10
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trnset = torchvision.datasets.CIFAR10(root='%s/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='%s/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            if self.dataset == 'CIFAR100':
                print(self.dataset)
                # mean and std are validated here: https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trnset = torchvision.datasets.CIFAR100(root='%s/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='%s/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=0)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'TinyImageNet':
                print(self.dataset)
                if 1:
                    download_tiny_imagenet(self.data_path)
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),  # Resize all images to 64x64
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.278, 0.276, 0.285])  # Common TinyImageNet stats
                ])
                # val_folder = os.path.join(self.data_path, 'Raw', 'tiny-imagenet-200', 'val')
                # restructure_val_split(val_folder)
                # tst_folder = os.path.join(self.data_path, 'Raw', 'tiny-imagenet-200', 'test')
                # restructure_val_split(tst_folder)
                # Tiny ImageNet typically has its structure as: train/class_x/*.JPEG and val/images/*.JPEG
                trnset = torchvision.datasets.ImageFolder(root='%s/Raw/tiny-imagenet-200/train' % self.data_path,
                                                        transform=transform)
                tstset = torchvision.datasets.ImageFolder(root='%s/Raw/tiny-imagenet-200/val' % self.data_path,
                                                        transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=len(trnset), shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=len(tstset), shuffle=False, num_workers=1)

                self.channels = 3
                self.width = 64
                self.height = 64
                self.n_cls = 200

            if self.dataset != 'emnist':
                trn_itr = trn_load.__iter__(); tst_itr = tst_load.__iter__() 
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
                tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)
            
            
            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

                # take first 10 classes of letters
                trn_idx = np.where(y_train < 10)[0]

                y_train = y_train[trn_idx]
                x_train = x_train[trn_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

                tst_idx = np.where(y_test < 10)[0]

                y_test = y_test[tst_idx]
                x_test = x_test[tst_idx]
                
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))
                
                # normalise train and test features

                trn_x = (x_train - mean_x) / std_x
                trn_y = y_train
                
                tst_x = (x_test  - mean_x) / std_x
                tst_y = y_test
                
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            # Shuffle Data
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]
            
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y
            
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            n_test_data_per_clnt = int((len(tst_y))/self.n_client)


            if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100' or self.dataset == 'TinyImageNet' or self.dataset == 'fmnist':
                # take some samples from client test data to create a server test set
                n_test_data_per_clnt = 0.8*n_test_data_per_clnt
                n_test_data_server = int(len(tst_y)-(self.n_client*n_test_data_per_clnt))
                

            server_test_data_list = (np.ones(1)*n_test_data_server).astype(int)
            # distributing server data sampling uniformly from the test data
            n_server = server_test_data_list[0]  # should equal 2000
            
            # exit()
            n_samples_per_class = n_server // self.n_cls
            print(n_server, n_samples_per_class, self.n_cls)
            # exit()
            server_indices = []
            # for cls in range(self.n_cls):
            #     cls_idx = np.where(tst_y.flatten() == cls)[0]
            #     print(f"Class {cls:3d}: {len(cls_idx):4d} samples")
            # exit()
            # For each class, randomly choose n_samples_per_class indices.
            for cls in range(self.n_cls):
                # Find all indices in tst_y for class 'cls'
                cls_idx = np.where(tst_y.flatten() == cls)[0]
                # print(len(cls_idx))
                # Randomly select n_samples_per_class indices from these
                selected = np.random.choice(cls_idx, size=n_samples_per_class, replace=False)
                server_indices.extend(selected)
            server_indices = np.array(server_indices)
            ###     
            
            if self.rule == 'Dirichlet':
            
                if self.dataset == 'CIFAR10':
                    rng = np.random.default_rng(seed=42)  
                    classes_group_1 = rng.choice(self.n_cls, 3, replace=False)
                    remaining_classes = np.setdiff1d(np.arange(self.n_cls), classes_group_1)
                    classes_group_2 = rng.choice(remaining_classes, 3, replace=False)
                    remaining_classes = np.setdiff1d(remaining_classes, classes_group_2)
                    classes_group_3 = remaining_classes

                    clients_group_1 = rng.choice(np.arange(self.n_client), 3, replace=False)
                    remaining_clients = np.setdiff1d(np.arange(self.n_client), clients_group_1)
                    clients_group_2 = rng.choice(remaining_clients, 3, replace=False)
                    remaining_clients = np.setdiff1d(remaining_clients, clients_group_2)
                    clients_group_3 = remaining_clients
                    print(f'{clients_group_1+1}: clients_group_1, {clients_group_2+1}: clients_group_2, {clients_group_3+1}: clients_group_3')
                    
                    if self.unbalanced_sgm != 0:
                        # Draw from lognormal distribution
                        # add the functionality of clnt test data for unbalanced samples
                        # unbalanced train, balanced test in number of total samples
                        clnt_test_data_list = (np.ones(self.n_client) * n_test_data_per_clnt).astype(int)
                        
                        # return the list of indices for the classes belonging to a certain group
                        idx_group1 = np.where(np.isin(trn_y, classes_group_1))[0]
                        idx_group2 = np.where(np.isin(trn_y, classes_group_2))[0]
                        idx_group3 = np.where(np.isin(trn_y, classes_group_3))[0]

                        clnt_data_list = np.zeros(self.n_client, dtype=int)

                        # Group 1
                        g1_counts = assign_unbalanced_to_group(
                            clients_group_1, len(idx_group1), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_1):
                            clnt_data_list[clnt_id] = g1_counts[i]

                        # Group 2
                        g2_counts = assign_unbalanced_to_group(
                            clients_group_2, len(idx_group2), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_2):
                            clnt_data_list[clnt_id] = g2_counts[i]

                        # Group 3
                        g3_counts = assign_unbalanced_to_group(
                            clients_group_3, len(idx_group3), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_3):
                            clnt_data_list[clnt_id] = g3_counts[i]

                        print("client sample counts:", clnt_data_list)
                    else:
                        clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
                        clnt_test_data_list = (np.ones(self.n_client) * n_test_data_per_clnt).astype(int)
                        
                    cls_priors = np.zeros((self.n_client, self.n_cls))
                    cls_priors_test = np.zeros((self.n_client, self.n_cls))

                    # Generate Dirichlet distributions for each group
                    dirichlet_priors_group_1 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_1), size=len(clients_group_1))
                    dirichlet_priors_group_2 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_2), size=len(clients_group_2))
                    dirichlet_priors_group_3 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_3), size=len(clients_group_3))
                    
                    dirichlet_test_priors_group_1 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_1), size=len(clients_group_1))
                    dirichlet_test_priors_group_2 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_2), size=len(clients_group_2))
                    dirichlet_test_priors_group_3 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_3), size=len(clients_group_3))
                    
                    # Assign priors for the client groups
                    for i, client in enumerate(clients_group_1):
                        for j, cls in enumerate(classes_group_1):
                            cls_priors[client, cls] = dirichlet_priors_group_1[i, j]

                    for i, client in enumerate(clients_group_2):
                        for j, cls in enumerate(classes_group_2):
                            cls_priors[client, cls] = dirichlet_priors_group_2[i, j]

                    for i, client in enumerate(clients_group_3):
                        for j, cls in enumerate(classes_group_3):
                            cls_priors[client, cls] = dirichlet_priors_group_3[i, j]

                    # Assign test priors for the client groups
                    for i, client in enumerate(clients_group_1):
                        for j, cls in enumerate(classes_group_1):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_1[i, j]

                    for i, client in enumerate(clients_group_2):
                        for j, cls in enumerate(classes_group_2):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_2[i, j]

                    for i, client in enumerate(clients_group_3):
                        for j, cls in enumerate(classes_group_3):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_3[i, j]

                    
                    #### override with existing data ####
                    cls_priors, cls_priors_test = get_cifar10_stored_data(self.rule_arg)
                    
                    plot_folder_path = f"./Client_Class_Distribution_Visual/{self.name}"
                    os.makedirs(plot_folder_path, exist_ok=True)

                    fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)
                    for client_idx, ax in enumerate(axes):
                        ax.bar(range(self.n_cls), cls_priors[client_idx], color=plt.cm.tab10.colors[:self.n_cls])
                        ax.set_title(f"Client {client_idx + 1}", fontsize=12)
                        ax.set_ylabel("Proportion")
                        ax.set_ylim(0, 1)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                    axes[-1].set_xlabel("Class Index")
                    fig.suptitle("Training Data Class Distributions Across Clients (Dirichlet Prior)", fontsize=16, y=1.02)
                    
                    plt.savefig("%s/inc_cifar10_train_class_prior_distribution_subgroups_3x3_dirichlet_0.2_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')

                    plt.close()

                    # Plot data for each client to visualize class prior distribution of client test data
                    fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)

                    for client_idx, ax in enumerate(axes):
                        ax.bar(range(self.n_cls), cls_priors_test[client_idx], color=plt.cm.tab10.colors[:self.n_cls])
                        ax.set_title(f"Client {client_idx + 1}", fontsize=12)
                        ax.set_ylabel("Proportion")
                        ax.set_ylim(0, 1)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                    axes[-1].set_xlabel("Class Index")
                    fig.suptitle("Test Data Class Distributions Across Clients (Dirichlet Prior)", fontsize=16, y=1.02)
                    
                    plt.savefig("%s/inc_cifar10_test_class_prior_distribution_subgroups_3x3_dirichlet_0.2_1.0.png"  %plot_folder_path, dpi=300, bbox_inches='tight')

                    plt.close()
                elif self.dataset == 'fmnist':
                    rng = np.random.default_rng(seed=42)  
                    #rng = np.random.default_rng(seed=0)
                    classes_group_1 = rng.choice(self.n_cls, 3, replace=False)
                    remaining_classes = np.setdiff1d(np.arange(self.n_cls), classes_group_1)
                    classes_group_2 = rng.choice(remaining_classes, 3, replace=False)
                    remaining_classes = np.setdiff1d(remaining_classes, classes_group_2)
                    classes_group_3 = remaining_classes

                    clients_group_1 = rng.choice(np.arange(self.n_client), 3, replace=False)
                    remaining_clients = np.setdiff1d(np.arange(self.n_client), clients_group_1)
                    clients_group_2 = rng.choice(remaining_clients, 3, replace=False)
                    remaining_clients = np.setdiff1d(remaining_clients, clients_group_2)
                    clients_group_3 = remaining_clients
                    print(f'{clients_group_1+1}: clients_group_1, {clients_group_2+1}: clients_group_2, {clients_group_3+1}: clients_group_3')
                    
                    if self.unbalanced_sgm != 0:
                        # Draw from lognormal distribution
                        # add the functionality of clnt test data for unbalanced samples
                        # unbalanced train, balanced test in number of total samples
                        clnt_test_data_list = (np.ones(self.n_client) * n_test_data_per_clnt).astype(int)
                        
                        # return the list of indices for the classes belonging to a certain group
                        idx_group1 = np.where(np.isin(trn_y, classes_group_1))[0]
                        idx_group2 = np.where(np.isin(trn_y, classes_group_2))[0]
                        idx_group3 = np.where(np.isin(trn_y, classes_group_3))[0]

                        clnt_data_list = np.zeros(self.n_client, dtype=int)

                        # Group 1
                        g1_counts = assign_unbalanced_to_group(
                            clients_group_1, len(idx_group1), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_1):
                            clnt_data_list[clnt_id] = g1_counts[i]

                        # Group 2
                        g2_counts = assign_unbalanced_to_group(
                            clients_group_2, len(idx_group2), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_2):
                            clnt_data_list[clnt_id] = g2_counts[i]

                        # Group 3
                        g3_counts = assign_unbalanced_to_group(
                            clients_group_3, len(idx_group3), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_3):
                            clnt_data_list[clnt_id] = g3_counts[i]

                        print("client sample counts:", clnt_data_list)
                        # breakpoint()
                        # clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
                        # clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
                        # diff = np.sum(clnt_data_list) - len(trn_y)

                        # # Add/Subtract the excess number starting from first client
                        # if diff!= 0:
                        #     for clnt_i in range(self.n_client):
                        #         if clnt_data_list[clnt_i] > diff:
                        #             clnt_data_list[clnt_i] -= diff
                        #             break
                        # print(clnt_data_list)

                    else:
                        clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
                        clnt_test_data_list = (np.ones(self.n_client) * n_test_data_per_clnt).astype(int)
                        
                    cls_priors = np.zeros((self.n_client, self.n_cls))
                    cls_priors_test = np.zeros((self.n_client, self.n_cls))

                    # Generate Dirichlet distributions for each group
                    dirichlet_priors_group_1 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_1), size=len(clients_group_1))
                    dirichlet_priors_group_2 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_2), size=len(clients_group_2))
                    dirichlet_priors_group_3 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_3), size=len(clients_group_3))
                    
                    dirichlet_test_priors_group_1 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_1), size=len(clients_group_1))
                    dirichlet_test_priors_group_2 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_2), size=len(clients_group_2))
                    dirichlet_test_priors_group_3 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_3), size=len(clients_group_3))
                    
                    # Assign priors for the client groups
                    for i, client in enumerate(clients_group_1):
                        for j, cls in enumerate(classes_group_1):
                            cls_priors[client, cls] = dirichlet_priors_group_1[i, j]

                    for i, client in enumerate(clients_group_2):
                        for j, cls in enumerate(classes_group_2):
                            cls_priors[client, cls] = dirichlet_priors_group_2[i, j]

                    for i, client in enumerate(clients_group_3):
                        for j, cls in enumerate(classes_group_3):
                            cls_priors[client, cls] = dirichlet_priors_group_3[i, j]

                    # Assign test priors for the client groups
                    for i, client in enumerate(clients_group_1):
                        for j, cls in enumerate(classes_group_1):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_1[i, j]

                    for i, client in enumerate(clients_group_2):
                        for j, cls in enumerate(classes_group_2):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_2[i, j]

                    for i, client in enumerate(clients_group_3):
                        for j, cls in enumerate(classes_group_3):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_3[i, j]

                    plot_folder_path = f"./Client_Class_Distribution_Visual/{self.name}"
                    os.makedirs(plot_folder_path, exist_ok=True)

                    fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)
                    for client_idx, ax in enumerate(axes):
                        ax.bar(range(self.n_cls), cls_priors[client_idx], color=plt.cm.tab10.colors[:self.n_cls])
                        ax.set_title(f"Client {client_idx + 1}", fontsize=12)
                        ax.set_ylabel("Proportion")
                        ax.set_ylim(0, 1)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                    axes[-1].set_xlabel("Class Index")
                    fig.suptitle("Training Data Class Distributions Across Clients (Dirichlet Prior)", fontsize=16, y=1.02)
                    
                    plt.savefig("%s/inc_fmnist_train_class_prior_distribution_subgroups_3x3_dirichlet_1.0_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')

                    plt.close()

                    # Plot data for each client to visualize class prior distribution of client test data
                    fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)

                    for client_idx, ax in enumerate(axes):
                        ax.bar(range(self.n_cls), cls_priors_test[client_idx], color=plt.cm.tab10.colors[:self.n_cls])
                        ax.set_title(f"Client {client_idx + 1}", fontsize=12)
                        ax.set_ylabel("Proportion")
                        ax.set_ylim(0, 1)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                    axes[-1].set_xlabel("Class Index")
                    fig.suptitle("Test Data Class Distributions Across Clients (Dirichlet Prior)", fontsize=16, y=1.02)
                    
                    plt.savefig("%s/inc_fmnist_test_class_prior_distribution_subgroups_3x3_dirichlet_1.0_1.0.png"  %plot_folder_path, dpi=300, bbox_inches='tight')

                    plt.close()
                elif self.dataset == 'CIFAR100':
                    rng = np.random.default_rng(seed=42)  
                    classes_group_1 = rng.choice(self.n_cls, 20, replace=False)
                    remaining_classes = np.setdiff1d(np.arange(self.n_cls), classes_group_1)
                    classes_group_2 = rng.choice(remaining_classes, 20, replace=False)
                    remaining_classes = np.setdiff1d(remaining_classes, classes_group_2)
                    classes_group_3 = rng.choice(remaining_classes, 20, replace=False)
                    remaining_classes = np.setdiff1d(remaining_classes, classes_group_3)
                    classes_group_4 = rng.choice(remaining_classes, 20, replace=False)
                    remaining_classes = np.setdiff1d(remaining_classes, classes_group_4)
                    classes_group_5 = remaining_classes

                    clients_group_1 = rng.choice(np.arange(self.n_client), 4, replace=False)
                    remaining_clients = np.setdiff1d(np.arange(self.n_client), clients_group_1)
                    clients_group_2 = rng.choice(remaining_clients, 4, replace=False)
                    remaining_clients = np.setdiff1d(remaining_clients, clients_group_2)
                    clients_group_3 = rng.choice(remaining_clients, 4, replace=False)
                    remaining_clients = np.setdiff1d(remaining_clients, clients_group_3)
                    clients_group_4 = rng.choice(remaining_clients, 4, replace=False)
                    remaining_clients = np.setdiff1d(remaining_clients, clients_group_4)
                    clients_group_5 = remaining_clients
                    print(f'{clients_group_1+1}: clients_group_1, {clients_group_2+1}: clients_group_2, {clients_group_3+1}: clients_group_3, {clients_group_4+1}: clients_group_4, {clients_group_5+1}: clients_group_5')
                    
                    if self.unbalanced_sgm != 0:
                        # Draw from lognormal distribution
                        # add the functionality of clnt test data for unbalanced samples
                        # unbalanced train, balanced test in number of total samples
                        clnt_test_data_list = (np.ones(self.n_client) * n_test_data_per_clnt).astype(int)
                        
                        # return the list of indices for the classes belonging to a certain group
                        idx_group1 = np.where(np.isin(trn_y, classes_group_1))[0]
                        idx_group2 = np.where(np.isin(trn_y, classes_group_2))[0]
                        idx_group3 = np.where(np.isin(trn_y, classes_group_3))[0]
                        idx_group4 = np.where(np.isin(trn_y, classes_group_4))[0]
                        idx_group5 = np.where(np.isin(trn_y, classes_group_5))[0]

                        clnt_data_list = np.zeros(self.n_client, dtype=int)

                        # Group 1
                        g1_counts = assign_unbalanced_to_group(
                            clients_group_1, len(idx_group1), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_1):
                            clnt_data_list[clnt_id] = g1_counts[i]

                        # Group 2
                        g2_counts = assign_unbalanced_to_group(
                            clients_group_2, len(idx_group2), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_2):
                            clnt_data_list[clnt_id] = g2_counts[i]

                        # Group 3
                        g3_counts = assign_unbalanced_to_group(
                            clients_group_3, len(idx_group3), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_3):
                            clnt_data_list[clnt_id] = g3_counts[i]

                        # Group 4
                        g4_counts = assign_unbalanced_to_group(
                            clients_group_4, len(idx_group4), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_4):
                            clnt_data_list[clnt_id] = g4_counts[i]

                        # Group 3
                        g5_counts = assign_unbalanced_to_group(
                            clients_group_5, len(idx_group5), self.unbalanced_sgm, rng=rng
                        )
                        for i, clnt_id in enumerate(clients_group_5):
                            clnt_data_list[clnt_id] = g5_counts[i]

                        print("client sample counts:", clnt_data_list)
                        # breakpoint()
                        # clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
                        # clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
                        # diff = np.sum(clnt_data_list) - len(trn_y)

                        # # Add/Subtract the excess number starting from first client
                        # if diff!= 0:
                        #     for clnt_i in range(self.n_client):
                        #         if clnt_data_list[clnt_i] > diff:
                        #             clnt_data_list[clnt_i] -= diff
                        #             break
                        # print(clnt_data_list)
                    else:
                        clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
                        clnt_test_data_list = (np.ones(self.n_client) * n_test_data_per_clnt).astype(int)
                        
                    # Initialize the class priors for all clients
                    cls_priors = np.zeros((self.n_client, self.n_cls))
                    cls_priors_test = np.zeros((self.n_client, self.n_cls))

                    # Generate Dirichlet distributions for each group
                    dirichlet_priors_group_1 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_1), size=len(clients_group_1))
                    dirichlet_priors_group_2 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_2), size=len(clients_group_2))
                    dirichlet_priors_group_3 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_3), size=len(clients_group_3))
                    dirichlet_priors_group_4 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_4), size=len(clients_group_4))
                    dirichlet_priors_group_5 = np.random.dirichlet(alpha=[self.rule_arg] * len(classes_group_5), size=len(clients_group_5))

                    dirichlet_test_priors_group_1 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_1), size=len(clients_group_1))
                    dirichlet_test_priors_group_2 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_2), size=len(clients_group_2))
                    dirichlet_test_priors_group_3 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_3), size=len(clients_group_3))
                    dirichlet_test_priors_group_4 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_4), size=len(clients_group_4))
                    dirichlet_test_priors_group_5 = np.random.dirichlet(alpha=[self.rule_arg_test] * len(classes_group_5), size=len(clients_group_5))

                    # Assign priors for the client groups
                    for i, client in enumerate(clients_group_1):
                        for j, cls in enumerate(classes_group_1):
                            cls_priors[client, cls] = dirichlet_priors_group_1[i, j]

                    for i, client in enumerate(clients_group_2):
                        for j, cls in enumerate(classes_group_2):
                            cls_priors[client, cls] = dirichlet_priors_group_2[i, j]

                    for i, client in enumerate(clients_group_3):
                        for j, cls in enumerate(classes_group_3):
                            cls_priors[client, cls] = dirichlet_priors_group_3[i, j]

                    for i, client in enumerate(clients_group_4):
                        for j, cls in enumerate(classes_group_4):
                            cls_priors[client, cls] = dirichlet_priors_group_4[i, j]

                    for i, client in enumerate(clients_group_5):
                        for j, cls in enumerate(classes_group_5):
                            cls_priors[client, cls] = dirichlet_priors_group_5[i, j]

                    # Assign test priors for the client groups
                    for i, client in enumerate(clients_group_1):
                        for j, cls in enumerate(classes_group_1):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_1[i, j]

                    for i, client in enumerate(clients_group_2):
                        for j, cls in enumerate(classes_group_2):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_2[i, j]

                    for i, client in enumerate(clients_group_3):
                        for j, cls in enumerate(classes_group_3):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_3[i, j]

                    for i, client in enumerate(clients_group_4):
                        for j, cls in enumerate(classes_group_4):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_4[i, j]

                    for i, client in enumerate(clients_group_5):
                        for j, cls in enumerate(classes_group_5):
                            cls_priors_test[client, cls] = dirichlet_test_priors_group_5[i, j]
                    # breakpoint()
                    # Plot data for each client to visualize class distribution of client training data

                    plot_folder_path = f"./Client_Class_Distribution_Visual/{self.name}"
                    os.makedirs(plot_folder_path, exist_ok=True)

                    fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)
                    for client_idx, ax in enumerate(axes):
                        ax.bar(range(self.n_cls), cls_priors[client_idx], color=plt.cm.tab10.colors[:self.n_cls])
                        ax.set_title(f"Client {client_idx + 1}", fontsize=12)
                        ax.set_ylabel("Proportion")
                        ax.set_ylim(0, 1)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                    axes[-1].set_xlabel("Class Index")
                    fig.suptitle("Training Data Class Distributions Across Clients (Dirichlet Prior)", fontsize=16, y=1.02)
                    
                    plt.savefig("%s/inc_cifar100_train_class_prior_distribution_subgroups_3x3_dirichlet_0.2_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')

                    plt.close()

                    # Plot data for each client to visualize class prior distribution of client test data
                    fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)

                    for client_idx, ax in enumerate(axes):
                        ax.bar(range(self.n_cls), cls_priors_test[client_idx], color=plt.cm.tab10.colors[:self.n_cls])
                        ax.set_title(f"Client {client_idx + 1}", fontsize=12)
                        ax.set_ylabel("Proportion")
                        ax.set_ylim(0, 1)
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                    axes[-1].set_xlabel("Class Index")
                    fig.suptitle("Test Data Class Distributions Across Clients (Dirichlet Prior)", fontsize=16, y=1.02)
                    
                    plt.savefig("%s/inc_cifar100_test_class_prior_distribution_subgroups_3x3_dirichlet_0.2_1.0.png"  %plot_folder_path, dpi=300, bbox_inches='tight')

                    plt.close()
                
                else:
                    clnt_data_list = (np.ones(self.n_client) * n_data_per_clnt).astype(int)
                    clnt_test_data_list = (np.ones(self.n_client) * n_test_data_per_clnt).astype(int)
                    cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                    cls_priors_test   = np.random.dirichlet(alpha=[self.rule_arg_test]*self.n_cls,size=self.n_client)

                # # breakpoint()

                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)] # indexation to find the indexes of the class instances in training data
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                prior_cumsum_test = np.cumsum(cls_priors_test, axis=1)
                idx_list_test = [np.where(tst_y==i)[0] for i in range(self.n_cls)] # indexation to find the indexes of the class instances in test cls data
                cls_amount_test = [len(idx_list_test[i]) for i in range(self.n_cls)]
                # removing the test samples from clients already sampled for the server
                for cls in range(self.n_cls):
                    idx_list_test[cls] = np.setdiff1d(idx_list_test[cls], server_indices) # remove the indexes assigned to the server
                    cls_amount_test[cls] = len(idx_list_test[cls])

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]

                clnt_x_test = [ np.zeros((clnt_test_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y_test = [ np.zeros((clnt_test_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]

                server_x = [ np.zeros((server_test_data_list[0], self.channels, self.height, self.width)).astype(np.float32) ]
                server_y = [ np.zeros((server_test_data_list[0], 1)).astype(np.int64) ]

                # assigning the server data
                server_x[0] = tst_x[server_indices]
                server_y[0] = tst_y[server_indices]

                # --------------------------------------------------------------
                class_idx_stacks = [idx_list[c].tolist() for c in range(self.n_cls)]

                for clnt in range(self.n_client):
                    n_needed = int(clnt_data_list[clnt])
                    if n_needed == 0:
                        continue

                    # --- 1. draw requested counts for every class -----------------
                    req_counts = np.random.multinomial(n_needed, cls_priors[clnt])

                    # --- 2. satisfy as much as possible ---------------------------
                    taken_counts = np.zeros(self.n_cls, dtype=int)
                    left_over    = 0
                    for c in range(self.n_cls):
                        take = min(req_counts[c], len(class_idx_stacks[c]))
                        taken_counts[c] = take
                        left_over      += req_counts[c] - take      # overflow to re‑distribute
                        # pop the indices we actually take
                        for _ in range(take):
                            idx = class_idx_stacks[c].pop()         # O(1) pop from list tail
                            clnt_x[clnt][n_needed - 1] = trn_x[idx] # fill from the back
                            clnt_y[clnt][n_needed - 1] = trn_y[idx]
                            n_needed -= 1

                    # --- 3. redistribute any overflow -----------------------------
                    if left_over > 0:
                        # Priority 1: Try to pull only from classes this client is allowed to have
                        still_available = [c for c in range(self.n_cls) if len(class_idx_stacks[c]) > 0 and cls_priors[clnt][c] > 0]
                        
                        for _ in range(left_over):
                            # Fallback: If allowed classes run completely dry, we MUST pull from global to satisfy the array quota
                            if len(still_available) == 0:
                                still_available = [c for c in range(self.n_cls) if len(class_idx_stacks[c]) > 0]
                            
                            # Absolute failsafe if the entire dataset is mysteriously empty
                            if len(still_available) == 0:
                                break
                                
                            c = np.random.choice(still_available)
                            idx = class_idx_stacks[c].pop()
                            clnt_x[clnt][n_needed - 1] = trn_x[idx]
                            clnt_y[clnt][n_needed - 1] = trn_y[idx]
                            n_needed -= 1
                            if len(class_idx_stacks[c]) == 0:       
                                still_available.remove(c)
                # --------------------------------------------------------------
                # breakpoint()

                # plotting the actual class distribution in client training data
                fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)
                for client_idx, ax in enumerate(axes):
                    labels = clnt_y[client_idx].flatten()
                    # Compute counts for each class from 0 to self.n_cls - 1.
                    counts = [np.sum(labels == cls) for cls in range(self.n_cls)]
                    total = np.sum(counts)
                    proportions = np.array(counts) / total if total > 0 else np.zeros(self.n_cls)
                    ax.bar(range(self.n_cls), proportions, color=plt.cm.tab10.colors[:self.n_cls])
                    ax.set_title(f"Client {client_idx + 1}, num_samples {len(clnt_x[client_idx])}", fontsize=12)
                    ax.set_ylabel("Proportion")
                    ax.set_ylim(0, 1)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)

                axes[-1].set_xlabel("Class Index")
                fig.suptitle("Actual Training Data Class Distributions Across Clients", fontsize=16, y=1.02)
                if self.dataset == 'CIFAR10':
                    plt.savefig("%s/inc_cifar10_actual_client_train_class_distribution_subgroups_3x3_dirichlet_0.5_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                elif self.dataset == 'CIFAR100':
                    plt.savefig("%s/inc_cifar100_actual_client_train_class_distribution_subgroups_3x3_dirichlet_0.5_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                elif self.dataset == 'TinyImageNet':
                    plt.savefig("%s/inc_Tinyimagenet_actual_client_train_class_distribution_subgroups_3x3_dirichlet_1.0_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                elif self.dataset == 'fmnist':
                    plt.savefig("%s/inc_fmnist_actual_client_train_class_distribution_subgroups_3x3_dirichlet_1.0_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                plt.close()

                class_idx_stacks_test = [idx_list_test[c].tolist() for c in range(self.n_cls)]

                for clnt in range(self.n_client):
                    n_needed = int(clnt_test_data_list[clnt])
                    if n_needed == 0:
                        continue

                    # --- 1. draw requested counts for every class -----------------
                    req_counts = np.random.multinomial(n_needed, cls_priors_test[clnt])

                    # --- 2. satisfy as much as possible ---------------------------
                    taken_counts = np.zeros(self.n_cls, dtype=int)
                    left_over    = 0
                    for c in range(self.n_cls):
                        take = min(req_counts[c], len(class_idx_stacks_test[c]))
                        taken_counts[c] = take
                        left_over      += req_counts[c] - take      # overflow to re‑distribute
                        # pop the indices we actually take
                        for _ in range(take):
                            idx = class_idx_stacks_test[c].pop()         # O(1) pop from list tail
                            clnt_x_test[clnt][n_needed - 1] = tst_x[idx] # fill from the back
                            clnt_y_test[clnt][n_needed - 1] = tst_y[idx]
                            n_needed -= 1

                    # --- 3. redistribute any overflow (TEST DATA) -----------------
                    if left_over > 0:
                        # Priority 1: Try to pull only from classes this client is allowed to have
                        still_available = [c for c in range(self.n_cls) if len(class_idx_stacks_test[c]) > 0 and cls_priors_test[clnt][c] > 0]
                        
                        for _ in range(left_over):
                            # Fallback: If allowed classes run completely dry, we MUST pull from global to satisfy the array quota
                            if len(still_available) == 0:
                                still_available = [c for c in range(self.n_cls) if len(class_idx_stacks_test[c]) > 0]
                            
                            # Absolute failsafe if the entire dataset is mysteriously empty
                            if len(still_available) == 0:
                                break
                                
                            c = np.random.choice(still_available)
                            idx = class_idx_stacks_test[c].pop()
                            clnt_x_test[clnt][n_needed - 1] = tst_x[idx]
                            clnt_y_test[clnt][n_needed - 1] = tst_y[idx]
                            n_needed -= 1
                            if len(class_idx_stacks_test[c]) == 0:       
                                still_available.remove(c)

                # plotting the actual class distribution in client test data
                fig, axes = plt.subplots(self.n_client, 1, figsize=(8, 20), sharex=True, constrained_layout=True)
                for client_idx, ax in enumerate(axes):
                    labels = clnt_y_test[client_idx].flatten()
                    counts = [np.sum(labels == cls) for cls in range(self.n_cls)]
                    total = np.sum(counts)
                    proportions = np.array(counts) / total if total > 0 else np.zeros(self.n_cls)
                    ax.bar(range(self.n_cls), proportions, color=plt.cm.tab10.colors[:self.n_cls])
                    ax.set_title(f"Client {client_idx + 1}", fontsize=12)
                    ax.set_ylabel("Proportion")
                    ax.set_ylim(0, 1)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)

                axes[-1].set_xlabel("Class Index")
                fig.suptitle("Actual Test Data Class Distributions Across Clients", fontsize=16, y=1.02)
                if self.dataset == 'CIFAR10':
                    plt.savefig("%s/inc_cifar10_actual_client_test_class_distribution_subgroups_3x3_dirichlet_0.5_1.0.png.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                elif self.dataset == 'CIFAR100':
                    plt.savefig("%s/inc_cifar100_actual_client_test_class_distribution_subgroups_3x3_dirichlet_0.5_1.0.png.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                elif self.dataset == 'TinyImageNet':
                    plt.savefig("%s/inc_Tinyimagenet_actual_client_test_class_distribution_subgroups_3x3_dirichlet_1.0_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                elif self.dataset == 'fmnist':
                    plt.savefig("%s/inc_fmnist_actual_client_test_class_distribution_subgroups_3x3_dirichlet_0.5_1.0.png" %plot_folder_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # training data for all clients
                clnt_x = np.asarray(clnt_x) 
                clnt_y = np.asarray(clnt_y)

                # test data for all clients
                clnt_x_test = np.asarray(clnt_x_test)
                clnt_y_test = np.asarray(clnt_y_test)

                server_x = np.asarray(server_x)
                server_y = np.asarray(server_y)
                # breakpoint()
                
                # deviation of client data distribution from class prior
                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from train data prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from train data prior: %.4f' %np.min(prior_real_diff))

                cls_means_test = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means_test[clnt,cls] = np.mean(clnt_y_test[clnt]==cls)
                prior_real_diff_test = np.abs(cls_means_test-cls_priors_test)
                print('--- Max deviation from test data prior: %.4f' %np.max(prior_real_diff_test))
                print('--- Min deviation from test data prior: %.4f' %np.min(prior_real_diff_test))

                #breakpoint()
            
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(trn_y)//100 % self.n_client == 0 
                # Only have the number clients if it divides 500
                # Perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx] # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        clnt_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = trn_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        clnt_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = trn_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            

            elif self.rule == 'iid' and self.dataset == 'CIFAR10' and self.unbalanced_sgm==0:           
    
                server_x = [ np.zeros((server_test_data_list[0], self.channels, self.height, self.width)).astype(np.float32) ]
                server_y = [ np.zeros((server_test_data_list[0], 1)).astype(np.int64) ]

                # assigning the server data
                server_x[0] = tst_x[server_indices]
                server_y[0] = tst_y[server_indices]

                # Perfect IID partitions for cifar10 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx] # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 10
                for i in range(self.n_client): # devices
                    for j in range(10): # class
                        clnt_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = trn_x[5000*j+n_cls_sample_per_device*i:5000*j+n_cls_sample_per_device*(i+1), :, :, :]
                        clnt_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = trn_y[5000*j+n_cls_sample_per_device*i:5000*j+n_cls_sample_per_device*(i+1), :] 
            
                # Partitioning client test data
                #remaining_idx = np.setdiff1d(np.arange(len(tst_y)), server_indices)
                #idx_tst = np.argsort(tst_y[remaining_idx, 0])

                #n_tst_data_per_clnt = len(idx_tst) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                #clnt_x_test = np.zeros((self.n_client, n_tst_data_per_clnt, 3, 32, 32), dtype=np.float32)
                #clnt_y_test = np.zeros((self.n_client, n_tst_data_per_clnt, 1), dtype=np.float32)
                #tst_x = tst_x[idx_tst] 
                #tst_y = tst_y[idx_tst]

                remaining_idx = np.setdiff1d(np.arange(len(tst_y)), server_indices)

                # Subset first
                tst_x_rem = tst_x[remaining_idx]
                tst_y_rem = tst_y[remaining_idx]

                # Then sort and assign
                idx_tst = np.argsort(tst_y_rem[:, 0])
                tst_x = tst_x_rem[idx_tst]
                tst_y = tst_y_rem[idx_tst]


                n_tst_cls_sample_per_device = n_tst_data_per_clnt // self.n_client

                clnt_x_test = np.zeros((self.n_client, n_tst_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y_test = np.zeros((self.n_client, n_tst_data_per_clnt, 1), dtype=np.float32)
                for i in range(self.n_client): # devices
                    for j in range(10): # class
                        clnt_x_test[i, n_tst_cls_sample_per_device*j:n_tst_cls_sample_per_device*(j+1), :, :, :] = tst_x[800*j+n_tst_cls_sample_per_device*i:800*j+n_tst_cls_sample_per_device*(i+1), :, :, :]
                        clnt_y_test[i, n_tst_cls_sample_per_device*j:n_tst_cls_sample_per_device*(j+1), :] = tst_y[800*j+n_tst_cls_sample_per_device*i:800*j+n_tst_cls_sample_per_device*(i+1), :] 
            
                print("===== Train Data Distribution per Client =====")
                for i in range(self.n_client):
                    labels_train = clnt_y[i].astype(int).flatten()
                    unique_labels, counts = np.unique(labels_train, return_counts=True)
                    print(f"Client {i} train distribution: {dict(zip(unique_labels, counts))}")

                print("===== Test Data Distribution per Client =====")
                for i in range(self.n_client):
                    labels_test = clnt_y_test[i].astype(int).flatten()
                    unique_labels, counts = np.unique(labels_test, return_counts=True)
                    print(f"Client {i} test distribution: {dict(zip(unique_labels, counts))}")
            
            elif self.rule == 'iid':
                                
                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
            
                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                
                
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

            
            self.clnt_x = clnt_x; self.clnt_y = clnt_y

            if self.rule == 'Dirichlet' and (self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100'  or self.dataset == 'fmnist'):
                self.clnt_tst_x = clnt_x_test; self.clnt_tst_y = clnt_y_test
                self.server_x = server_x; self.server_y = server_y
            elif self.rule == 'iid' and (self.dataset == 'CIFAR10' or self.dataset == 'fmnist'):
                self.clnt_tst_x = clnt_x_test; self.clnt_tst_y = clnt_y_test
                self.server_x = server_x; self.server_y = server_y
            else:
                self.tst_x  = tst_x;  self.tst_y  = tst_y
            
            # Save data
            os.mkdir('%s/%s' %(self.data_path, self.name))
            
            if self.rule == 'Dirichlet' and (self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100' or self.dataset == 'fmnist'):
                np.save('%s/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x)
                np.save('%s/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y)

                np.save('%s/%s/clnt_tst_x.npy'  %(self.data_path, self.name),  clnt_x_test)
                np.save('%s/%s/clnt_tst_y.npy'  %(self.data_path, self.name),  clnt_y_test)

                np.save('%s/%s/server_x.npy'  %(self.data_path, self.name),  server_x)
                np.save('%s/%s/server_y.npy'  %(self.data_path, self.name),  server_y)
            elif self.rule == 'iid' and (self.dataset == 'CIFAR10' or self.dataset == 'fmnist'):
                np.save('%s/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x)
                np.save('%s/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y)

                np.save('%s/%s/clnt_tst_x.npy'  %(self.data_path, self.name),  clnt_x_test)
                np.save('%s/%s/clnt_tst_y.npy'  %(self.data_path, self.name),  clnt_y_test)

                np.save('%s/%s/server_x.npy'  %(self.data_path, self.name),  server_x)
                np.save('%s/%s/server_y.npy'  %(self.data_path, self.name),  server_y)
            else:
                np.save('%s/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x)
                np.save('%s/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y)

                np.save('%s/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
                np.save('%s/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)

        else:
            print("Data is already downloaded in the folder.")
            if self.rule == 'Dirichlet' and (self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100' or self.dataset == 'TinyImageNet' or self.dataset == 'fmnist'):
                self.clnt_x = np.load('%s/%s/clnt_x.npy' %(self.data_path, self.name), allow_pickle=True)
                self.clnt_y = np.load('%s/%s/clnt_y.npy' %(self.data_path, self.name), allow_pickle=True)
                self.n_client = len(self.clnt_x)

                self.clnt_tst_x  = np.load('%s/%s/clnt_tst_x.npy'  %(self.data_path, self.name), allow_pickle=True)
                self.clnt_tst_y  = np.load('%s/%s/clnt_tst_y.npy'  %(self.data_path, self.name), allow_pickle=True)

                self.server_x  = np.load('%s/%s/server_x.npy'  %(self.data_path, self.name), allow_pickle=True)
                self.server_y  = np.load('%s/%s/server_y.npy'  %(self.data_path, self.name), allow_pickle=True)  

            elif self.rule == 'iid' and (self.dataset == 'CIFAR10' or self.dataset == 'TinyImageNet' or self.dataset == 'fmnist'):
                self.clnt_x = np.load('%s/%s/clnt_x.npy' %(self.data_path, self.name), allow_pickle=True)
                self.clnt_y = np.load('%s/%s/clnt_y.npy' %(self.data_path, self.name), allow_pickle=True)
                self.n_client = len(self.clnt_x)

                self.clnt_tst_x  = np.load('%s/%s/clnt_tst_x.npy'  %(self.data_path, self.name), allow_pickle=True)
                self.clnt_tst_y  = np.load('%s/%s/clnt_tst_y.npy'  %(self.data_path, self.name), allow_pickle=True)

                self.server_x  = np.load('%s/%s/server_x.npy'  %(self.data_path, self.name), allow_pickle=True)
                self.server_y  = np.load('%s/%s/server_y.npy'  %(self.data_path, self.name), allow_pickle=True)                

            else:
                self.clnt_x = np.load('%s/%s/clnt_x.npy' %(self.data_path, self.name), allow_pickle=True)
                self.clnt_y = np.load('%s/%s/clnt_y.npy' %(self.data_path, self.name), allow_pickle=True)
                self.n_client = len(self.clnt_x)

                self.tst_x  = np.load('%s/%s/tst_x.npy'  %(self.data_path, self.name), allow_pickle=True)
                self.tst_y  = np.load('%s/%s/tst_y.npy'  %(self.data_path, self.name), allow_pickle=True)
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'TinyImageNet':
                self.channels = 3
                self.width = 64
                self.height = 64
                self.n_cls = 200
            if self.dataset == 'fashion_mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'fmnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
                
        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt + 
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')
        
        # print("      Test: " + 
        #     ', '.join(["%.3f" %np.mean(self.tst_y==cls) for cls in range(self.n_cls)]) + 
        #     ', Amount:%d' %self.tst_y.shape[0])


def assign_unbalanced_to_group(group_clients, total_samples_for_group, unbalanced_sigma, rng):
    """
    Distributes 'total_samples_for_group' among 'group_clients' via lognormal,
    then do a rounding fix so the sum is exactly total_samples_for_group.
    Returns an array sample_counts of length len(group_clients).
    """
    n_gc = len(group_clients)
    if n_gc == 0:
        return np.array([], dtype=int)

    logn_draw = rng.lognormal(mean=np.log(total_samples_for_group / n_gc),
                              sigma=unbalanced_sigma,
                              size=n_gc)

    sample_counts = (logn_draw / np.sum(logn_draw) * total_samples_for_group).astype(int)
    
    diff = np.sum(sample_counts) - total_samples_for_group
    if diff != 0:
        for i in range(n_gc):
            if (sample_counts[i] > diff):
                sample_counts[i] -= diff
                break
    return sample_counts


def generate_syn_logistic(dimension, n_clnt, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):
    
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points
    
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)
    
    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_clnt)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' %np.sum(samples_per_user))
    
    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_clnt))
    data_y = list(range(n_clnt))

    mean_W = np.random.normal(0, alpha, n_clnt)
    B = np.random.normal(0, beta, n_clnt)

    mean_x = np.zeros((n_clnt, dimension))

    if not iid_dat: # If IID then make all 0s.
        for i in range(n_clnt):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))
    
    if iid_sol: # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))
    
    for i in range(n_clnt):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1,1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y
        
class DatasetSynthetic:
    def __init__(self, alpha, beta, theta, iid_sol, iid_data, n_dim, n_clnt, n_cls, avg_data, name_prefix):
        self.dataset = 'synt'
        self.name  = name_prefix + '_'
        self.name += '%d_%d_%d_%d_%f_%f_%f_%s_%s' %(n_dim, n_clnt, n_cls, avg_data,
                alpha, beta, theta, iid_sol, iid_data)

        data_path = 'Data'
        if (not os.path.exists('%s/%s/' %(data_path, self.name))):
            # Generate data
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_clnt=n_clnt, n_cls=n_cls, avg_data=avg_data, 
                                        alpha=alpha, beta=beta, theta=theta, 
                                        iid_sol=iid_sol, iid_dat=iid_data)
            os.mkdir('%s/%s/' %(data_path, self.name))
            np.save('%s/%s/data_x.npy' %(data_path, self.name), data_x)
            np.save('%s/%s/data_y.npy' %(data_path, self.name), data_y)
        else:
            # Load data
            print('Load')
            data_x = np.load('%s/%s/data_x.npy' %(data_path, self.name), allow_pickle=True)
            data_y = np.load('%s/%s/data_y.npy' %(data_path, self.name), allow_pickle=True)

        for clnt in range(n_clnt):
            print(', '.join(['%.4f' %np.mean(data_y[clnt]==t) for t in range(n_cls)]))

        self.clnt_x = data_x
        self.clnt_y = data_y

        self.tst_x = np.concatenate(self.clnt_x, axis=0)
        self.tst_y = np.concatenate(self.clnt_y, axis=0)
        self.n_client = len(data_x)
        print(self.clnt_x.shape)

# Original prepration is from LEAF paper...
# This loads Shakespeare dataset only.
# data_path/train and data_path/test are assumed to be processed
# To make the dataset smaller,
# We take 2000 datapoints for each client in the train_set

class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')
        
        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Ignore groups information, combine test cases for different clients into one test data
        # Change structure to DatasetObject structure
        
        self.users = users[:10]
        print('numbers of users',len(self.users))
        # print(train_data['0']['x'][0:3],'x:y',train_data['0']['y'][:3],len(train_data['0']['y']),len(train_data['0']['x']))
        # print(test_data)
        # exit()

        self.n_client = len(self.users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        self.clnt_tst_x = list(range(self.n_client))
        self.clnt_tst_y = list(range(self.n_client))
        tst_data_count = 0
        Ta = crop_amount//tst_ratio
        # print(Ta)
        for clnt in range(self.n_client):
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(train_data[users[clnt]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[clnt]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[clnt]]['y'])[start:start+crop_amount]

            start_tst = np.random.randint(len(test_data[users[clnt]]['x'])-Ta)
            self.clnt_tst_x[clnt] = np.asarray(test_data[users[clnt]]['x'])[start_tst:start_tst+Ta]
            self.clnt_tst_y[clnt] = np.asarray(test_data[users[clnt]]['y'])[start_tst:start_tst+Ta]
            

        
        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        # print(tst_data_count)
        # exit()
        self.server_x = list(range(tst_data_count))
        self.server_y = list(range(tst_data_count))
        
        tst_data_count = 0
        for clnt in range(self.n_client):
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[clnt]]['x'])-curr_amount)
            self.server_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['x'])[start:start+curr_amount]
            self.server_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[clnt]]['y'])[start:start+curr_amount]
            
            tst_data_count += curr_amount
        # print(tst_data_count)
        # exit()
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        self.clnt_tst_x = np.asarray(self.clnt_tst_x)
        self.clnt_tst_y = np.asarray(self.clnt_tst_y)
  
        
        self.server_x = np.asarray(self.server_x)
        self.server_y = np.asarray(self.server_y)
        
        # Convert characters to numbers
        
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)

        self.clnt_tst_x_char = np.copy(self.clnt_tst_x)
        self.clnt_tst_y_char = np.copy(self.clnt_tst_y)
        
        self.server_x_char = np.copy(self.server_x)
        self.server_y_char = np.copy(self.server_y)
        
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))

        self.clnt_tst_x = list(range(len(self.clnt_tst_x_char)))
        self.clnt_tst_y = list(range(len(self.clnt_tst_x_char)))
        

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        for clnt in range(len(self.clnt_tst_x_char)):
            clnt_tst_list_x = list(range(len(self.clnt_tst_x_char[clnt])))
            clnt_tst_list_y = list(range(len(self.clnt_tst_x_char[clnt])))
            
            for idx in range(len(self.clnt_tst_x_char[clnt])):
                clnt_tst_list_x[idx] = np.asarray(word_to_indices(self.clnt_tst_x_char[clnt][idx]))
                clnt_tst_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_tst_y_char[clnt][idx]))).reshape(-1)

            self.clnt_tst_x[clnt] = np.asarray(clnt_tst_list_x)
            self.clnt_tst_y[clnt] = np.asarray(clnt_tst_list_y)
                
        self.clnt_tst_x = np.asarray(self.clnt_tst_x)
        self.clnt_tst_y = np.asarray(self.clnt_tst_y)
        
        # print(self.clnt_tst_x.shape) 
        # print(self.clnt_x.shape)
        # exit()
        self.server_x = list(range(len(self.server_x_char)))
        self.server_y = list(range(len(self.server_x_char)))
        
        for idx in range(len(self.server_x_char)):
            self.server_x[idx] = np.asarray(word_to_indices(self.server_x_char[idx]))
            self.server_y[idx] = np.argmax(np.asarray(letter_to_vec(self.server_y_char[idx]))).reshape(-1)
        # print('clnt test',len(self.server_x[0]))
        # exit()
        self.server_x = np.asarray(self.server_x)
        self.server_y = np.asarray(self.server_y)
    
        
        
class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=10, crop_amount=2000, tst_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.        
        # Change structure to DatasetObject structure
        
        # self.users = users 
        self.users = users
        print('number of users before sampling', len(users))
        # exit()
        tst_data_count_per_clnt = (crop_amount//tst_ratio)
        # Group clients that have at least crop_amount datapoints
        arr = []
        for clnt in range(len(users)):
            if (len(np.asarray(train_data[users[clnt]]['y'])) > crop_amount 
                and len(np.asarray(test_data[users[clnt]]['y'])) > 1*tst_data_count_per_clnt):
                arr.append(clnt)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]
        # print(self.user_idx)
        # exit()
        self.clnt_x = list(range(self.n_client))
        self.clnt_y = list(range(self.n_client))
        self.clnt_tst_x = list(range(self.n_client))
        self.clnt_tst_y = list(range(self.n_client))

        tst_data_count = 0
        Ta = crop_amount//tst_ratio
        # print(Ta, len(test_data[users[clnt]]['x'])-Ta )
        # exit()
        for clnt, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + clnt)
            print(Ta, len(test_data[users[idx]]['x']) )
            start = np.random.randint(len(train_data[users[idx]]['x'])-crop_amount)
            self.clnt_x[clnt] = np.asarray(train_data[users[idx]]['x'])[start:start+crop_amount]
            self.clnt_y[clnt] = np.asarray(train_data[users[idx]]['y'])[start:start+crop_amount]

            start_tst = np.random.randint(len(test_data[users[idx]]['x'])-Ta)
            self.clnt_tst_x[clnt] = np.asarray(test_data[users[idx]]['x'])[start_tst:start_tst+Ta]
            self.clnt_tst_y[clnt] = np.asarray(test_data[users[idx]]['y'])[start_tst:start_tst+Ta]
            
        tst_data_count = (crop_amount//tst_ratio) * self.n_client
        # self.tst_x = list(range(tst_data_count))
        # self.tst_y = list(range(tst_data_count))
        
        self.server_x = list(range(tst_data_count))
        self.server_y = list(range(tst_data_count))

        tst_data_count = 0

        for clnt, idx in enumerate(self.user_idx):
            
            curr_amount = (crop_amount//tst_ratio)
            np.random.seed(rand_seed + clnt)
            start = np.random.randint(len(test_data[users[idx]]['x'])-curr_amount)
            # self.tst_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            # self.tst_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
            self.server_x[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            self.server_y[tst_data_count: tst_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
            tst_data_count += curr_amount
            
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)
        
        # self.tst_x = np.asarray(self.tst_x)
        # self.tst_y = np.asarray(self.tst_y)

        self.clnt_tst_x = np.asarray(self.clnt_tst_x)
        self.clnt_tst_y = np.asarray(self.clnt_tst_y)
  
        
        self.server_x = np.asarray(self.server_x)
        self.server_y = np.asarray(self.server_y)
        
        # # Convert characters to numbers
        
        # self.clnt_x_char = np.copy(self.clnt_x)
        # self.clnt_y_char = np.copy(self.clnt_y)
        
        # self.tst_x_char = np.copy(self.tst_x)
        # self.tst_y_char = np.copy(self.tst_y)
        
        # self.clnt_x = list(range(len(self.clnt_x_char)))
        # self.clnt_y = list(range(len(self.clnt_x_char)))
        

        # for clnt in range(len(self.clnt_x_char)):
        #     clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
        #     clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
        #     for idx in range(len(self.clnt_x_char[clnt])):
        #         clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
        #         clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

        #     self.clnt_x[clnt] = np.asarray(clnt_list_x)
        #     self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        # self.clnt_x = np.asarray(self.clnt_x)
        # self.clnt_y = np.asarray(self.clnt_y)
        
        
        # self.tst_x = list(range(len(self.tst_x_char)))
        # self.tst_y = list(range(len(self.tst_x_char)))
                
        # for idx in range(len(self.tst_x_char)):
        #     self.tst_x[idx] = np.asarray(word_to_indices(self.tst_x_char[idx]))
        #     self.tst_y[idx] = np.argmax(np.asarray(letter_to_vec(self.tst_y_char[idx]))).reshape(-1)
        
        # self.tst_x = np.asarray(self.tst_x)
        # self.tst_y = np.asarray(self.tst_y)
        #  Convert characters to numbers
        
        self.clnt_x_char = np.copy(self.clnt_x)
        self.clnt_y_char = np.copy(self.clnt_y)

        self.clnt_tst_x_char = np.copy(self.clnt_tst_x)
        self.clnt_tst_y_char = np.copy(self.clnt_tst_y)
        
        self.server_x_char = np.copy(self.server_x)
        self.server_y_char = np.copy(self.server_y)
        
        self.clnt_x = list(range(len(self.clnt_x_char)))
        self.clnt_y = list(range(len(self.clnt_x_char)))

        self.clnt_tst_x = list(range(len(self.clnt_tst_x_char)))
        self.clnt_tst_y = list(range(len(self.clnt_tst_x_char)))
        

        for clnt in range(len(self.clnt_x_char)):
            clnt_list_x = list(range(len(self.clnt_x_char[clnt])))
            clnt_list_y = list(range(len(self.clnt_x_char[clnt])))
            
            for idx in range(len(self.clnt_x_char[clnt])):
                clnt_list_x[idx] = np.asarray(word_to_indices(self.clnt_x_char[clnt][idx]))
                clnt_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_y_char[clnt][idx]))).reshape(-1)

            self.clnt_x[clnt] = np.asarray(clnt_list_x)
            self.clnt_y[clnt] = np.asarray(clnt_list_y)
                
        self.clnt_x = np.asarray(self.clnt_x)
        self.clnt_y = np.asarray(self.clnt_y)

        for clnt in range(len(self.clnt_tst_x_char)):
            clnt_tst_list_x = list(range(len(self.clnt_tst_x_char[clnt])))
            clnt_tst_list_y = list(range(len(self.clnt_tst_x_char[clnt])))
            
            for idx in range(len(self.clnt_tst_x_char[clnt])):
                clnt_tst_list_x[idx] = np.asarray(word_to_indices(self.clnt_tst_x_char[clnt][idx]))
                clnt_tst_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.clnt_tst_y_char[clnt][idx]))).reshape(-1)

            self.clnt_tst_x[clnt] = np.asarray(clnt_tst_list_x)
            self.clnt_tst_y[clnt] = np.asarray(clnt_tst_list_y)
                
        self.clnt_tst_x = np.asarray(self.clnt_tst_x)
        self.clnt_tst_y = np.asarray(self.clnt_tst_y)
        
        # print(self.clnt_tst_x.shape) 
        # print(self.clnt_x.shape)
        # exit()
        self.server_x = list(range(len(self.server_x_char)))
        self.server_y = list(range(len(self.server_x_char)))
        
        for idx in range(len(self.server_x_char)):
            self.server_x[idx] = np.asarray(word_to_indices(self.server_x_char[idx]))
            self.server_y[idx] = np.argmax(np.asarray(letter_to_vec(self.server_y_char[idx]))).reshape(-1)
        # print('clnt test',len(self.server_x[0]))
        # exit()
        self.server_x = np.asarray(self.server_x)
        self.server_y = np.asarray(self.server_y)
    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist' or self.name == 'fmnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
            
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == 'TinyImageNet':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
        
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
                
        elif self.name == 'shakespeare':
            
            self.X_data = data_x
            self.y_data = data_y
                
            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()
            
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist' or self.name == 'fmnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == 'TinyImageNet':
            img = self.X_data[idx]
            if self.name == 'TinyImageNet':
                H = 64
                W = 64
            else:
                H = 32
                W = 32
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping 
                    pad = 4 if H == 32 else 8
                    extended_img = np.zeros((3,H + pad *2, W + pad *2)).astype(np.float32)
                    # extended_img[:,pad:-pad,pad:-pad] = img
                    extended_img[:, pad:pad+H, pad:pad+W] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+H,dim_2:dim_2+W]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
            
        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx] 
            return x, y
            

def download_tiny_imagenet(dest_path):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        zip_path = os.path.join(dest_path, 'tiny-imagenet-200.zip')
        extract_path = os.path.join(dest_path, 'Raw')
        target_dir = os.path.join(extract_path, 'tiny-imagenet-200')

        if not os.path.isdir(target_dir):
            os.makedirs(extract_path, exist_ok=True)
            print("Downloading Tiny ImageNet...")
            urllib.request.urlretrieve(url, zip_path)
            print("Extracting Tiny ImageNet...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_path)
            os.remove(zip_path)
            print("Done.")
        else:
            print("Tiny ImageNet already present.")

def restructure_val_split(val_root):
        images_dir = os.path.join(val_root, 'images')
        ann_file   = os.path.join(val_root, 'val_annotations.txt')

        # read filename → class mapping
        mapping = {}
        with open(ann_file, 'r') as f:
            for line in f:
                img, cls, *_ = line.strip().split('\t')
                mapping[img] = cls

        # make one folder per class
        for cls in set(mapping.values()):
            os.makedirs(os.path.join(val_root, cls), exist_ok=True)

        # move each image into its class folder
        for img, cls in mapping.items():
            src = os.path.join(images_dir, img)
            dst = os.path.join(val_root, cls, img)
            shutil.move(src, dst)

        # remove now-empty images dir
        shutil.rmtree(images_dir)
        print("Restructured val split into class subfolders.")
        
        
def get_cifar10_stored_data(rule_arg = '0.5'):
    
    if rule_arg == '0.5':
        train = np.array([
            [0.00, 0.32, 0.00, 0.00, 0.00, 0.04, 0.00, 0.00, 0.64, 0.00],  # C1  (group B)
            [0.00, 0.00, 0.27, 0.17, 0.47, 0.00, 0.00, 0.09, 0.00, 0.00],  # C2  (group C)
            [0.00, 0.00, 0.19, 0.02, 0.13, 0.00, 0.00, 0.66, 0.00, 0.00],  # C3  (group C)
            [0.00, 0.01, 0.00, 0.00, 0.00, 0.17, 0.00, 0.00, 0.82, 0.00],  # C4  (group B)
            [0.33, 0.00, 0.00, 0.00, 0.00, 0.00, 0.56, 0.00, 0.00, 0.11],  # C5  (group A)
            [0.00, 0.00, 0.00, 0.28, 0.71, 0.00, 0.00, 0.01, 0.00, 0.00],  # C6  (group C)
            [0.00, 0.85, 0.00, 0.00, 0.00, 0.08, 0.00, 0.00, 0.07, 0.00],  # C7  (group B)
            [0.63, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.07],  # C8  (group A)
            [0.79, 0.00, 0.00, 0.00, 0.00, 0.00, 0.18, 0.00, 0.00, 0.03],  # C9  (group A)
            [0.00, 0.00, 0.00, 0.02, 0.93, 0.00, 0.00, 0.05, 0.00, 0.00],  # C10 (group C)
            ], dtype=np.float64)
 
        test = np.array([
            [0.00, 0.58, 0.00, 0.00, 0.00, 0.42, 0.00, 0.00, 0.00, 0.00],  # C1  (group B)
            [0.00, 0.00, 0.27, 0.12, 0.13, 0.00, 0.00, 0.51, 0.00, 0.00],  # C2  (group C)
            [0.00, 0.00, 0.37, 0.47, 0.03, 0.00, 0.00, 0.14, 0.00, 0.00],  # C3  (group C)
            [0.00, 0.87, 0.00, 0.00, 0.00, 0.11, 0.00, 0.00, 0.02, 0.00],  # C4  (group B)
            [0.44, 0.00, 0.00, 0.00, 0.00, 0.00, 0.45, 0.00, 0.00, 0.11],  # C5  (group A)
            [0.00, 0.00, 0.48, 0.29, 0.17, 0.00, 0.00, 0.06, 0.00, 0.00],  # C6  (group C)
            [0.00, 0.76, 0.00, 0.00, 0.00, 0.09, 0.00, 0.00, 0.15, 0.00],  # C7  (group B)
            [0.81, 0.00, 0.00, 0.00, 0.00, 0.00, 0.17, 0.00, 0.00, 0.02],  # C8  (group A)
            [0.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.63, 0.00, 0.00, 0.30],  # C9  (group A)
            [0.00, 0.00, 0.02, 0.52, 0.23, 0.00, 0.00, 0.23, 0.00, 0.00],  # C10 (group C)
            ], dtype=np.float64)
    else:
        train = np.array([
                    [0.00, 0.38, 0.00, 0.00, 0.00, 0.17, 0.00, 0.00, 0.45, 0.00], # C1
                    [0.00, 0.00, 0.35, 0.19, 0.26, 0.00, 0.00, 0.20, 0.00, 0.00], # C2
                    [0.00, 0.00, 0.28, 0.31, 0.20, 0.00, 0.00, 0.21, 0.00, 0.00], # C3
                    [0.00, 0.24, 0.00, 0.00, 0.00, 0.53, 0.00, 0.00, 0.23, 0.00], # C4
                    [0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.41, 0.00, 0.00, 0.19], # C5
                    [0.00, 0.00, 0.06, 0.33, 0.33, 0.00, 0.00, 0.28, 0.00, 0.00], # C6
                    [0.00, 0.39, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.31, 0.00], # C7
                    [0.09, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.41], # C8
                    [0.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.00, 0.00, 0.40], # C9
                    [0.00, 0.00, 0.32, 0.17, 0.21, 0.00, 0.00, 0.30, 0.00, 0.00], # C10
                ],dtype=np.float64)
                
        test = np.array([
                    [0.00, 0.43, 0.00, 0.00, 0.00, 0.32, 0.00, 0.00, 0.25, 0.00], # C1
                    [0.00, 0.00, 0.19, 0.04, 0.29, 0.00, 0.00, 0.48, 0.00, 0.00], # C2
                    [0.00, 0.00, 0.31, 0.44, 0.23, 0.00, 0.00, 0.02, 0.00, 0.00], # C3
                    [0.00, 0.38, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.42, 0.00], # C4
                    [0.69, 0.00, 0.00, 0.00, 0.00, 0.00, 0.28, 0.00, 0.00, 0.03], # C5
                    [0.00, 0.00, 0.22, 0.13, 0.20, 0.00, 0.00, 0.45, 0.00, 0.00], # C6
                    [0.00, 0.20, 0.00, 0.00, 0.00, 0.48, 0.00, 0.00, 0.32, 0.00], # C7
                    [0.16, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.54], # C8
                    [0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.43, 0.00, 0.00, 0.42], # C9
                    [0.00, 0.00, 0.28, 0.39, 0.28, 0.00, 0.00, 0.05, 0.00, 0.00], # C10
                ],dtype=np.float64)
 

    train_norm = train / train.sum(axis=1, keepdims=True)
    test_norm  = test  / test.sum(axis=1,  keepdims=True)
    
    return train_norm, test_norm            