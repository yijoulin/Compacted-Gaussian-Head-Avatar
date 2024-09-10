import os
import pdb
from tqdm import tqdm
import time

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import defaultdict

#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#                     佛祖保佑 永無BUG
#
class compact_3D():
    def __init__(self, num_clusters=4096, num_iters=5):
        self.num_clusters = num_clusters
        self.num_kmeans_iters = num_iters
        self.nn_index = torch.empty(0)
        self.centers = torch.empty(0)
        self.vec_dim = 0
        self.cluster_ids = torch.empty(0,dtype=torch.long)
        self.cls_ids = torch.empty(0)
        self.excl_clusters = []
        self.excl_cluster_ids = []
        self.cluster_len = torch.empty(0)
        self.max_cnt = 0
        self.n_excl_cls = 0
        self.name_type = ""

        #==============================SELF-DEFINED===============================
        self.codebook = {'scale': torch.zeros(num_clusters, self.vec_dim)}

    def get_dist(self, x, y, mode='sq_euclidean'):
        """Calculate distance between all vectors in x and all vectors in y.

        x: (m, dim)
        y: (n, dim)
        dist: (m, n)
        """
        if mode == 'sq_euclidean_chunk':
            step = 65536
            if x.shape[0] < step:
                step = x.shape[0]
            dist = []
            for i in range(np.ceil(x.shape[0] / step).astype(int)):
                dist.append(torch.cdist(x[(i*step): (i+1)*step, :].unsqueeze(0), y.unsqueeze(0))[0])
            dist = torch.cat(dist, 0)
        elif mode == 'sq_euclidean':
            dist = torch.cdist(x.unsqueeze(0).detach(), y.unsqueeze(0).detach())[0]
        return dist

        # Update centers in non-cluster assignment iters using cached nn indices.
    def update_centers(self, feat):
        # scale
        feat = feat.detach().reshape(-1, self.vec_dim)
        # Update all clusters except the excluded ones in a single operation
        # Add a dummy element with zeros at the end
        feat = torch.cat([feat, torch.zeros_like(feat[:1]).cuda()], 0)
        self.centers = torch.sum(feat[self.cluster_ids, :].reshape(
            self.num_clusters, self.max_cnt, -1), dim=1)
        if len(self.excl_cluster_ids) > 0:
            for i, cls in enumerate(self.excl_clusters):
                # Division by num_points in cluster is done during the one-shot averaging of all
                # clusters below. Only the extra elements in the bigger clusters are added here.
                self.centers[cls] += torch.sum(feat[self.excl_cluster_ids[i], :], dim=0)
        self.centers /= (self.cluster_len + 1e-6)

        #save the updating centers to codebook
        #self.save_codebook()
    
    

    # Update centers during cluster assignment using mask matrix multiplication
    # Mask is obtained from distance matrix
    def update_centers_(self, feat, cluster_mask=None, nn_index=None, avg=False):
        feat = feat.detach().reshape(-1, self.vec_dim)
        centers = (cluster_mask.T @ feat)
        if avg:
            self.centers /= counts.unsqueeze(-1)
        return centers
    
    def equalize_cluster_size(self):
        """Make the size of all the clusters the same by appending dummy elements.

        """
        # Find the maximum number of elements in a cluster, make size of all clusters
        # equal by appending dummy elements until size is equal to size of max cluster.
        # If max is too large, exclude it and consider the next biggest. Use for loop for
        # the excluded clusters and a single operation for the remaining ones for
        # updating the cluster centers.

        unq, n_unq = torch.unique(self.nn_index, return_counts=True)
        # Find max cluster size and exclude clusters greater than a threshold
        topk = 100
        if len(n_unq) < topk:
            topk = len(n_unq)
        max_cnt_topk, topk_idx = torch.topk(n_unq, topk)
        self.max_cnt = max_cnt_topk[0]
        idx = 0
        self.excl_clusters = []
        self.excl_cluster_ids = []
        while(self.max_cnt > 5000):
            self.excl_clusters.append(unq[topk_idx[idx]])
            idx += 1
            if idx < topk:
                self.max_cnt = max_cnt_topk[idx]
            else:
                break
        self.n_excl_cls = len(self.excl_clusters)
        self.excl_clusters = sorted(self.excl_clusters)
        # Store the indices of elements for each cluster
        all_ids = []
        cls_len = []
        for i in range(self.num_clusters):
            cur_cluster_ids = torch.where(self.nn_index == i)[0]
            # For excluded clusters, use only the first max_cnt elements
            # for averaging along with other clusters. Separately average the
            # remaining elements just for the excluded clusters.
            cls_len.append(torch.Tensor([len(cur_cluster_ids)]))
            if i in self.excl_clusters:
                self.excl_cluster_ids.append(cur_cluster_ids[self.max_cnt:])
                cur_cluster_ids = cur_cluster_ids[:self.max_cnt]
            # Append dummy elements to have same size for all clusters
            all_ids.append(torch.cat([cur_cluster_ids, -1 * torch.ones((self.max_cnt - len(cur_cluster_ids)),
                                                                       dtype=torch.long).cuda()]))
        all_ids = torch.cat(all_ids).type(torch.long)
        cls_len = torch.cat(cls_len).type(torch.long)
        self.cluster_ids = all_ids
        self.cluster_len = cls_len.unsqueeze(1).cuda()
        self.cls_ids = self.nn_index

    def cluster_assign(self, feat, feat_scaled=None):
        # 這邊Feat 是一種tensor
        # quantize with kmeans
        feat = feat.detach() # .detach() 生成的新张量不再跟踪计算历史，因此不会参与反向传播
        feat = feat.reshape(-1, self.vec_dim)
        if feat_scaled is None:
            feat_scaled = feat
            scale = feat[0] / (feat_scaled[0] + 1e-8)
        if len(self.centers) == 0:
            self.centers = feat[torch.randperm(feat.shape[0])[:self.num_clusters], :]

        # start kmeans
        chunk = True
        counts = torch.zeros(self.num_clusters, dtype=torch.float32).cuda() + 1e-6
        centers = torch.zeros_like(self.centers)
        for iteration in range(self.num_kmeans_iters):
            # chunk for memory issues
            if chunk:
                self.nn_index = None
                i = 0
                chunk = 10000
                while True:
                    dist = self.get_dist(feat[i*chunk:(i+1)*chunk, :], self.centers)
                    curr_nn_index = torch.argmin(dist, dim=-1)
                    # Assign a single cluster when distance to multiple clusters is same
                    dist = F.one_hot(curr_nn_index, self.num_clusters).type(torch.float32)
                    curr_centers = self.update_centers_(feat[i*chunk:(i+1)*chunk, :], dist, curr_nn_index, avg=False)
                    counts += dist.detach().sum(0) + 1e-6
                    centers += curr_centers
                    if self.nn_index == None:
                        self.nn_index = curr_nn_index
                    else:
                        self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                    i += 1
                    if i*chunk > feat.shape[0]:
                        break

            self.centers = centers / counts.unsqueeze(-1)
            # Reinitialize to 0
            centers[centers != 0] = 0.
            counts[counts > 0.1] = 0.

        if chunk:
            self.nn_index = None
            i = 0
            # chunk = 100000
            while True:
                dist = self.get_dist(feat_scaled[i * chunk:(i + 1) * chunk, :], self.centers)
                curr_nn_index = torch.argmin(dist, dim=-1)
                if self.nn_index == None:
                    self.nn_index = curr_nn_index
                else:
                    self.nn_index = torch.cat((self.nn_index, curr_nn_index), dim=0)
                i += 1
                if i * chunk > feat.shape[0]:
                    break
        self.equalize_cluster_size()
    
    def rescale(self, feat, scale=None):
        """Scale the feature to be in the range [-1, 1] by dividing by its max value.

        """
        if scale is None:
            return feat / (abs(feat).max(dim=0)[0] + 1e-8)
        else:
            return feat / (scale + 1e-8)
        
################################################################################################
########################################SELF-DEFINED-PARAMETERS#################################
################################################################################################
    def initializing(self,type_name):
        self.name_type = type_name
        if self.name_type == "scale":
            self.vec_dim = 3
        if self.name_type == "rotation":
            self.vec_dim = 4
        if self.name_type == "scale_rot":
            self.vec_dim = 7
        if self.name_type == "feature":
            self.vec_dim = 128

    def get_vec_dim(self):
        return self.vec_dim
    
    def get_nn_index(self):
        return self.nn_index

    def get_centers(self):
        # CENTERS 就是codebook阿
        return self.centers

    
    def forward_scale_rot(self, scale, rotation, assign=False):
        """Combine both scaling and rotation for a single k-Means"""
        feat_scaled = torch.cat([self.rescale(scale), self.rescale(rotation)], 1)
        feat = torch.cat([scale, rotation], 1)
        if assign:
            self.cluster_assign(feat, feat_scaled)
        else:
            self.update_centers(feat)
    
    def forward_feature(self, feature, assign=False):
        if assign:
            self.cluster_assign(feature)
        else:
            self.update_centers(feature)
 
    def print_info(self):
        #print(self.num_clusters) #400
        #print(self.num_kmeans_iters) #10
        print(self.nn_index) #tensor([228,  17, 314,  ...,  26,  71,  71], device='cuda:0')
        print(self.centers.size()) 
        print(self.centers)
        #tensor([[-4.3968, -4.3610, -4.4018],
        #[-4.8601, -4.8911, -4.8195],
        #[-5.0242, -5.0899, -5.0829],
        #...,
        #[-4.8080, -4.7671, -4.7898],
        #[-4.9678, -4.9390, -4.9351],
        #[-4.4037, -4.2577, -4.3653]], device='cuda:0')
       # print(self.vec_dim) #3
        #print(self.cluster_ids) #tensor([ 892, 1381, 1702,  ...,   -1,   -1,   -1], device='cuda:0')
        print("=====================================")
        #print(self.cls_ids)  #tensor([228,  17, 314,  ...,  26,  71,  71], device='cuda:0')
        #print(self.excl_clusters) #[]
        #print(self.excl_cluster_ids) #[]
        print(self.cluster_len.size()) 
        #tensor([[ 411],
        #[ 365],
        #[ 182],
        #[ 334],
        #[ 319],
        #...
        #[ 520],
        #[ 506],
        #[ 132]], device='cuda:0')
        print(self.max_cnt) #tensor(1523, device='cuda:0')
        #
        print(self.n_excl_cls) #0
    