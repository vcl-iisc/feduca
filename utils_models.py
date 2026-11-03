from utils_libs import *
import torchvision.models as models
#from transformers import AutoTokenizer
#from transformers import AutoModel




class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
        if self.name == 'Linear':
            [self.n_dim, self.n_out] = args
            self.fc = nn.Linear(self.n_dim, self.n_out)
          
        if self.name == 'mnist' or self.name == 'fmnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, self.n_cls)
            
        if self.name == 'emnist':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
        
        if self.name == 'cifar10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'cifar100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) 
            self.fc3 = nn.Linear(192, self.n_cls)
            
        if self.name == 'Resnet18':
            resnet18 = models.resnet18(weights=True)
            # resnet18.fc = nn.Linear(512, 10)
            resnet18.fc = nn.Linear(512, 100) # for CIFAR100
            #resnet18.fc = nn.Linear(512, 200)
            # Change BN to GN 
            resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
            self.model = resnet18


        if self.name == 'shakespeare':
            bundle_path = "../sst5_federated_bundle.pt"
            checkpoint = torch.load(bundle_path, weights_only=False)
            glove_matrix = checkpoint.get('glove_matrix')

            # 2. Set dimensions based on the GloVe matrix
            if glove_matrix is not None:
                vocab_size = glove_matrix.shape[0]   # 30522
                embedding_dim = glove_matrix.shape[1] # 300
                self.embedding = nn.Embedding.from_pretrained(glove_matrix, freeze=False)
                #print("here")
                #exit(0)
            else:
                # Fallback if the bundle doesn't have GloVe
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                vocab_size = len(tokenizer)
                embedding_dim = 128 
                self.embedding = nn.Embedding(vocab_size, embedding_dim)

            hidden_size =300
            num_LSTM = 1
            self.n_cls = 5 #20
            num_layers = 2

            # 3. Initialize LSTM with the CORRECT embedding_dim
            #self.stacked_GRU = nn.GRU(
            #    input_size=embedding_dim, 
            #    hidden_size=hidden_size, 
            #    num_layers=num_layers, 
            #    bidirectional=True, 
            #    batch_first=True
            #)
            
            # The Attention Layer
            #self.attention = nn.Linear(2 * hidden_size, 1)
            #
            #self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(2 * hidden_size, self.n_cls)
            self.stacked_LSTM = nn.LSTM(
               input_size=embedding_dim, 
               hidden_size=hidden_size, 
               num_layers=num_LSTM, 
               bidirectional=True, 
               batch_first=True
            )
            #
            #

            self.attention = nn.Linear(2 * hidden_size, 1)
            
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(2 * hidden_size, self.n_cls)
              
        
    def forward(self, x):
        if self.name == 'Linear':
            x = self.fc(x)
            
        if self.name == 'mnist' or self.name == 'fmnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
  
        if self.name == 'emnist':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'cifar100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
        if self.name == 'Resnet18':
            x = self.model(x)
        
        if self.name == 'shakespeare':
            # x shape: [batch_size, seq_len]
            embedded = self.embedding(x)
            
            # out shape: [batch_size, seq_len, 2 * hidden_size]
            # We ignore h_n and c_n now, because we use the whole sequence 'out'
            out, _ = self.stacked_LSTM(embedded)
            #out, _ = self.stacked_GRU(embedded)
            
            # --- APPLY ATTENTION ---
            # 1. Score each word in the sequence
            # attn_weights shape: [batch_size, seq_len, 1]
            attn_weights = self.attention(out)
            
            # 2. Normalize scores to probabilities using Softmax
            # Words with high sentiment value will get scores closer to 1.0
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # 3. Multiply the LSTM outputs by their attention weights and sum them up
            # context_vector shape: [batch_size, 2 * hidden_size]
            context_vector = torch.sum(attn_weights * out, dim=1)
            
            # Apply dropout to the finalized context vector
            context_dropped = self.dropout(context_vector)
            
            # Pass to classifier
            x = self.fc(context_dropped)
            
        return x
    