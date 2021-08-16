import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        layers = []
        sizes = ([(input_depth, filter_size)] + 
                 [(filter_size, filter_size)]*(len(layer_config)-2) + 
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        return x

class Latent(nn.Module):
    def __init__(self, hidden_dim = 256, filter_dim = 512):
        super(Latent, self).__init__()
        self.mean = PositionwiseFeedForward(hidden_dim, filter_dim, hidden_dim,
                                                                 layer_config='lll', padding = 'left',
                                                                 dropout=0)
        self.var = PositionwiseFeedForward(hidden_dim, filter_dim, hidden_dim,
                                                                 layer_config='lll', padding = 'left', 
                                                                 dropout=0)
        self.mean_p = PositionwiseFeedForward(hidden_dim*2, filter_dim, hidden_dim,
                                                                 layer_config='lll', padding = 'left', 
                                                                 dropout=0)
        self.var_p = PositionwiseFeedForward(hidden_dim*2, filter_dim, hidden_dim,
                                                                 layer_config='lll', padding = 'left', 
                                                                 dropout=0)

    def forward(self,x,x_p, train=True, fixed_z=False, seed=None):
        mean = self.mean(x)
        log_var = self.var(x)

        if seed is not None:
            torch.manual_seed(seed)
        eps = torch.randn(x.size())
        std = torch.exp(0.5 * log_var)
        eps = eps.to(x.device)

        if fixed_z:
            z = mean
        else:
            #print('Randomness Here')
            z = eps * std + mean

        kld_loss = 0
        if x_p is not None:
            mean_p = self.mean_p(torch.cat((x_p,x),dim=-1))
            log_var_p = self.var_p(torch.cat((x_p,x),dim=-1))
            kld_loss = self.gaussian_kld(mean_p,log_var_p,mean,log_var)
            kld_loss = torch.mean(kld_loss)

        if train:
            std = torch.exp(0.5 * log_var_p)
            eps = eps.to(x.device)
            z = eps * std + mean_p
        return kld_loss, z
    
    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=-1)
        return kld