import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import pairwise_distances, csls_sim

class MutualVAE(nn.Module):

    def __init__(self, in_dim, hidden_dims, latent_dim=None, **kwargs):
        super(MutualVAE, self).__init__()

        if latent_dim:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = hidden_dims[-1]

        modules = []

        # encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):

        x = self.encoder(x)

        mu_x = self.fc_mu(x)
        log_var_x = self.fc_var(x)

        return (mu_x, log_var_x)

    def decode(self, z, reparameterize=False):
        if reparameterize:
            z = self.reparameterize(*z)

        z = self.decoder_input(z)
        x = self.decoder(z)

        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.rand_like(std)
        return eps * std + mu

    def forward(self, embs, train_links, left_ents, right_ents):

        # train_links (x,y) used for flows : x->y and y->x, supervised learning
        # left_ents, right_ents used for flows : x->x and y->y, self-supervised learning

        # flows: x->y and y->x
        x = embs[train_links[:, 0]]
        y = embs[train_links[:, 1]]

        z_xy, z_yx = self.encode(x), self.encode(y)
        y_xy, x_yx = self.decode(z_xy, reparameterize=True), self.decode(
            z_yx, reparameterize=True)

        # flows : x->x and y->y
        sampled_x, sampled_y = embs[left_ents], embs[right_ents]
        z_xx, z_yy = self.encode(sampled_x), self.encode(sampled_y)
        x_xx, y_yy = self.decode(z_xx, reparameterize=True), self.decode(
            z_yy, reparameterize=True)

        flows = {'xx': (sampled_x, z_xx, x_xx),
                 'yy': (sampled_y, z_yy, y_yy),
                 'xy': (x, z_xy, y_xy),
                 'yx': (y, z_yx, x_yx)}

        return flows

class NeighborDecoder(nn.Module):
    def __init__(self, sub_dim, ent_embs) -> None:
        super().__init__()

        self.ent_embs = None
        self.subdecoder = nn.Sequential(nn.Linear(sub_dim, sub_dim),
                                               nn.Tanh(),
                                               nn.Dropout(0.5),
                                               nn.BatchNorm1d(sub_dim),
                                               nn.Linear(sub_dim, sub_dim),
                                               nn.Tanh(),
                                               nn.Dropout(0.5),
                                               nn.BatchNorm1d(sub_dim),
                                            )
        self.register_parameter('bias', nn.Parameter(torch.zeros(ent_embs.shape[0])))

    def forward(self, x):
        output = self.subdecoder(x)
        output = x @ self.ent_embs.T + self.bias
        return F.tanh(output)



class GEEA(nn.Module):

    def __init__(self, args, kgs, concrete_features, sub_dims, joint_dim, ent_embs, fusion_layer):
        super().__init__()
        self.args = args
        self.kgs = kgs
        self.latent_dim=sub_dims[0]

        self.subgenerators = []
        self.subdecoders = []

        self.num_none_concrete_feature = 0
        for i, sub_dim, concrete_feature in zip(range(len(sub_dims)), sub_dims, concrete_features):
            if concrete_feature is not None:
                subgenerator = MutualVAE(in_dim=sub_dim,
                                        hidden_dims=[sub_dim,]*args.num_layers,
                                        latent_dim=sub_dim)
                self.subgenerators.append(subgenerator)
                
                if i==-1:
                    subdecoder = NeighborDecoder(sub_dim, ent_embs)
                else:
                    subdecoder = nn.Sequential(nn.Linear(sub_dim, 1000),
                                               nn.Tanh(),
                                               nn.Dropout(0.5),
                                               nn.BatchNorm1d(1000),
                                               nn.Linear(1000, concrete_feature.shape[-1]),
                                            )
                self.subdecoders.append(subdecoder)
            else:
                self.num_none_concrete_feature += 1
                
        
        


        self.subgenerators = nn.ModuleList(self.subgenerators)
        self.subdecoders = nn.ModuleList(self.subdecoders)

        # for distribtuion matching
        self.sample_prop = 1./7
        self.number_samples = int(
            len(self.kgs['left_ents']) * self.sample_prop)

        # for prior and post reconstruction
        self.prior_reconstruction_loss_func = nn.BCEWithLogitsLoss()
        self.post_reconstruction_loss_func = nn.MSELoss()
        self.concrete_features = concrete_features
        
        self.fusion_layer = fusion_layer

        # xx, yy, xy, yx
        self.flow_weights = [1, 1, 1, 1]

    def distribution_match_loss(self, outputs):

        def kld_loss(mu, logvar, kld_weight=self.sample_prop):
            return kld_weight * torch.mean(-.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)

        # output = (x, z=(mu, var), reconstrctued_x)
        xx_distribution_match_loss = [
            kld_loss(*output['xx'][1]) for output in outputs]
        yy_distribution_match_loss = [
            kld_loss(*output['yy'][1]) for output in outputs]

        return sum(xx_distribution_match_loss) + sum(yy_distribution_match_loss) 

    
    def sampled_bce_loss(self, predicted, label, neg_ratio=5):
        pos_mask = torch.where(label>0)


        neg = torch.randint(high=label.shape[-1], size=(len(pos_mask[0])*neg_ratio,))
        neg_mask = [pos_mask[0].repeat(neg_ratio), neg]

        predicted_pos = predicted[pos_mask]
        label_pos = torch.ones_like(predicted_pos)
        predicted_neg = predicted[neg_mask]
        label_neg = torch.zeros_like(predicted_neg)

        loss = self.prior_reconstruction_loss_func(predicted_pos, label_pos) + self.prior_reconstruction_loss_func(predicted_neg, label_neg) / neg_ratio
        return loss
    
    def sampled_crossentropy_loss(self, predicted, label, neg_ratio=1):
        pos_mask, labels = torch.where(label>0)
        sampled = torch.randperm(len(pos_mask))[:3500]
        pos_mask, labels = pos_mask[sampled], labels[sampled]
        
        predicted_pos = predicted[pos_mask]
        
        loss = F.cross_entropy(predicted_pos.cuda(), labels.cuda()) 
        return loss
    
    
    def prior_reconstruction_loss(self, outputs, train_links, left_ents, right_ents):

        prior_reconstruction_loss = []
        
        for output, subdecoder, concrete_feature in  zip(outputs, self.subdecoders, self.concrete_features):
            
            
            
            reconstructed_xx = subdecoder(output['xx'][-1])
            reconstructed_yy = subdecoder(output['yy'][-1])
            reconstructed_xy = subdecoder(output['xy'][-1])
            reconstructed_yx = subdecoder(output['yx'][-1])
            
            
            concrete_xx = concrete_feature[left_ents].cuda()
            concrete_yy = concrete_feature[right_ents].cuda()
            concrete_xy = concrete_feature[train_links[:, 1]].cuda()
            concrete_yx = concrete_feature[train_links[:, 0]].cuda()

 
            loss_xx = self.prior_reconstruction_loss_func(
                reconstructed_xx, concrete_xx)
            loss_yy = self.prior_reconstruction_loss_func(
                reconstructed_yy, concrete_yy)
            loss_xy = self.prior_reconstruction_loss_func(
                reconstructed_xy, concrete_xy)
            loss_yx = self.prior_reconstruction_loss_func(
                reconstructed_yx, concrete_yx)

            loss_list = [loss_xx, loss_yy, loss_xy, loss_yx]

            prior_reconstruction_loss += [sum(loss*flow_weight for loss, flow_weight in zip(loss_list, self. flow_weights)), ]
         
        return sum(prior_reconstruction_loss)

    def re_fusion(self, sub_embs):
        sub_embs = sub_embs+[None,]*self.num_none_concrete_feature
        return self.fusion_layer(*sub_embs)
    
    def reconstruction_loss(self, outputs):
        loss = 0.
        for output in outputs:
            for flow in output.keys():
                input_, z, output_ = output[flow]
                loss += self.post_reconstruction_loss_func(input_.detach(), output_)
        return loss
            
        

    def post_reconstruction_loss(self, outputs, joint_emb, train_links, left_ents, right_ents):

        xx, yy, xy, yx = [], [], [], []

        for output, subdecoder in zip(outputs, self.subdecoders):
            xx.append(output['xx'][-1])
            yy.append(output['yy'][-1])
            xy.append(output['xy'][-1])
            yx.append(output['yx'][-1])

        # reconstructed
        reconstructed_xx = self.re_fusion(xx)
        reconstructed_yy = self.re_fusion(yy)
        reconstructed_xy = self.re_fusion(xy)
        reconstructed_yx = self.re_fusion(yx)

        # the targets
        joint_emb = joint_emb.detach()
        joint_xx = joint_emb[left_ents]
        joint_yy = joint_emb[right_ents]
        joint_xy = joint_emb[train_links[:, 1]]
        joint_yx = joint_emb[train_links[:, 0]]

        # loss
        loss_xx = self.post_reconstruction_loss_func(
            reconstructed_xx, joint_xx) 
        loss_yy = self.post_reconstruction_loss_func(
            reconstructed_yy, joint_yy) 
        loss_xy = self.post_reconstruction_loss_func(
            reconstructed_xy, joint_xy) 
        loss_yx = self.post_reconstruction_loss_func(
            reconstructed_yx, joint_yx) 

        return loss_xx + loss_yy + loss_xy + loss_yx
    

    def encode(self, xs, sub_embs):
        sub_embs = [embs for embs in sub_embs if embs is not None]
        
        x_zs = [subgenerator.encode(embs[xs])
                   for embs, subgenerator in zip(sub_embs, self.subgenerators)]
        
        return x_zs

    def decode(self, zs, reparameterize=False):

        reconstructed_x = [subgenerator.decode(z, reparameterize=reparameterize)
                   for subgenerator, z in zip(self.subgenerators, zs)]

        return reconstructed_x
    
    def sample(self, num):
        z = torch.randn(num, self.latent_dim).cuda()
        
        samples = self.decode(z)
        
        return samples
    
    def id2feature(self):
        pass


    def sample_from_x_to_y(self, xs, sub_embs):
        zs = self.encode(xs, sub_embs)

        samples = self.decode(zs, reparameterize=True)

        return samples
    


    def forward(self, train_links, sub_embs, joint_emb):
        sub_embs = [embs for embs in sub_embs if embs is not None]
        self.subdecoders[0].ent_embs=sub_embs[0]

        # for self-supervised learning
        left_ents = np.random.choice(
            self.kgs['left_ents'], self.number_samples, replace=False)
        right_ents = np.random.choice(
            self.kgs['right_ents'], self.number_samples, replace=False)

        outputs = [subgenerator(embs, train_links, left_ents, right_ents)
                   for embs, subgenerator in zip(sub_embs, self.subgenerators)]

        distribution_match_loss = self.distribution_match_loss(outputs)
        prior_reconstruction_loss = self.prior_reconstruction_loss(
            outputs, train_links, left_ents, right_ents)
        post_reconstruction_loss = self.post_reconstruction_loss(outputs, joint_emb, train_links, left_ents, right_ents)
        
        reconstruction_loss = self.reconstruction_loss(outputs)

        print('DistMatch Loss: %.3f; PriorRec Loss: %.3f; PostRec Loss: %.3f' % (
            distribution_match_loss.item(), prior_reconstruction_loss.item(), post_reconstruction_loss.item()+reconstruction_loss.item()))
        
        return distribution_match_loss*0.5 + prior_reconstruction_loss + reconstruction_loss + post_reconstruction_loss