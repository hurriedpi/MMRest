import math
import pdb
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


def _with_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
	return t.to(device) if t.device != device else t


def _chol_solve(M: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	d = M.shape[0]
	jitter_vals = (0.0, 1e-8, 1e-7, 1e-6, 1e-5)
	I = torch.eye(d, dtype=M.dtype, device=M.device)
	for j in jitter_vals:
		try:
			L = torch.linalg.cholesky(M + j * I)

			X = torch.cholesky_solve(B, L)
			return X
		except RuntimeError:
			continue

	return torch.linalg.pinv(M) @ B


class PyMMLCNLoss(nn.Module):



	def __init__(self, v1: float = 0.1, v2: float = 1.0, alpha_: float = 0.5, beta: float = 1e-2,
				 max_clusters: int = 7, max_pairs_per_cluster: int = 4000,
				 random_state: int = 0,
				 xi: float = 1.0):
		super().__init__()
		self.v1 = float(v1)
		self.v2 = float(v2)
		self.alpha_ = float(alpha_)  
		self.beta = float(beta)  

		self.max_clusters = int(max_clusters)
		self.max_pairs_per_cluster = int(max_pairs_per_cluster)
		self.random_state = int(random_state)

		self.xi = float(xi)


		self._inited = False
		self._feat_dim = None

		self.M_0 = None    
		self.Delta_M = None 


	@staticmethod
	def _label_bin(v: np.ndarray) -> np.ndarray:
		bins = np.empty_like(v, dtype=np.int64)

		bins.fill(3)

		bins[(v >= -3.0) & (v <= -2.5)] = 0
		bins[(v > -2.5) & (v <= -1.5)] = 1
		bins[(v > -1.5) & (v <= -0.5)] = 2
		bins[(v > -0.5) & (v < 0.5)] = 3
		bins[(v >= 0.5) & (v < 1.5)] = 4
		bins[(v >= 1.5) & (v < 2.5)] = 5
		bins[(v >= 2.5) & (v <= 3.0)] = 6
		return bins

	@staticmethod
	def _kmeans_cpu(X: np.ndarray, k: int, random_state: int = 0, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
		n, d = X.shape
		if n == 0:
			return np.empty((0,), dtype=np.int64), np.empty((0, d))
		k_eff = max(1, min(k, n))

		try:
			from sklearn.cluster import KMeans  
			km = KMeans(n_clusters=k_eff, n_init=10, max_iter=max_iter, random_state=random_state)
			labels = km.fit_predict(X)
			centers = km.cluster_centers_
			return labels.astype(np.int64), centers.astype(np.float32)
		except Exception:
			pass

		rng = np.random.RandomState(random_state)

		centers = np.empty((k_eff, d), dtype=X.dtype)
		idx0 = rng.randint(0, n)
		centers[0] = X[idx0]
		d2 = ((X - centers[0]) ** 2).sum(axis=1)
		for i in range(1, k_eff):
			probs = d2 / (d2.sum() + 1e-12)
			idx = rng.choice(n, p=probs)
			centers[i] = X[idx]

			d2 = np.minimum(d2, ((X - centers[i]) ** 2).sum(axis=1))
		labels = np.zeros((n,), dtype=np.int64)
		for _ in range(max_iter):

			dist2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n, k)
			new_labels = dist2.argmin(axis=1)
			if np.array_equal(new_labels, labels):
				break
			labels = new_labels

			for j in range(k_eff):
				mask = (labels == j)
				if not np.any(mask):

					centers[j] = X[rng.randint(0, n)]
				else:
					centers[j] = X[mask].mean(axis=0)
		return labels.astype(np.int64), centers.astype(np.float32)

	@staticmethod
	def _nearest_receptive(X: np.ndarray, centers: np.ndarray) -> List[np.ndarray]:

		if centers.shape[0] == 0:
			return []
		d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n, k)
		idx = d2.argmin(axis=1)  # (n,)
		groups: List[List[int]] = [[] for _ in range(centers.shape[0])]
		for i, g in enumerate(idx.tolist()):
			groups[g].append(i)
		return [np.asarray(g, dtype=np.int64) for g in groups]

	def _merge_by_category(self, centers: np.ndarray, R_cells: List[np.ndarray],
						   label_bins: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:

		if len(R_cells) == 0:
			return np.empty((0, centers.shape[1])), [], []

		center_bins: List[int] = []
		for inds in R_cells:
			if inds.size == 0:
				center_bins.append(-1)
				continue
			bins = label_bins[inds]
			vals, cnts = np.unique(bins, return_counts=True)
			center_bins.append(int(vals[cnts.argmax()]))


		cat_to_center_idx: Dict[int, List[int]] = {}
		for ci, cat in enumerate(center_bins):
			if cat < 0:
				continue
			cat_to_center_idx.setdefault(cat, []).append(ci)

		present_cats = sorted(cat_to_center_idx.keys())
		merged_centers: List[np.ndarray] = []
		merged_R: List[np.ndarray] = []
		for cat in present_cats:
			idxs = cat_to_center_idx[cat]

			merged_centers.append(centers[idxs].mean(axis=0))

			if len(idxs) == 1:
				merged_R.append(R_cells[idxs[0]])
			else:
				merged = np.unique(np.concatenate([R_cells[j] for j in idxs], axis=0))
				merged_R.append(merged)

		return np.stack(merged_centers, axis=0), merged_R, present_cats


	def _ensure_params(self, feat_dim: int, device: torch.device):
		if self._inited and self._feat_dim == feat_dim:
			return
		self._feat_dim = int(feat_dim)
		Kmax, D = self.max_clusters, self._feat_dim
		

		M_0 = torch.zeros((D, D), dtype=torch.float32)
		M_0.fill_(0.0)
		M_0.diagonal().fill_(1.0)
		self.M_0 = nn.Parameter(M_0)
		

		Delta_M = torch.zeros((Kmax, D, D), dtype=torch.float32)
		for k in range(Kmax):
			Delta_M[k].fill_(0.0)
			Delta_M[k].diagonal().fill_(1.0)
		self.Delta_M = nn.Parameter(Delta_M)
		
		self._inited = True
		self.to(device)

	def _build_pairs(self, R_cells: List[np.ndarray], labels_ref: np.ndarray,
					 K: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		S_list: List[np.ndarray] = []
		D_list: List[np.ndarray] = []
		for i in range(K):
			inds = R_cells[i]
			m = inds.size
			if m < 2:
				S_list.append(np.empty((0, 2), dtype=np.int64))
				D_list.append(np.empty((0, 2), dtype=np.int64))
				continue
			ii, jj = np.triu_indices(m, 1)
			pairs = np.stack([inds[ii], inds[jj]], axis=1)
			same = (labels_ref[pairs[:, 0]] == labels_ref[pairs[:, 1]])
			S = pairs[same]
			D = pairs[~same]

			if S.shape[0] > self.max_pairs_per_cluster:
				S = S[np.random.RandomState(self.random_state).choice(S.shape[0], self.max_pairs_per_cluster, replace=False)]
			if D.shape[0] > self.max_pairs_per_cluster:
				D = D[np.random.RandomState(self.random_state).choice(D.shape[0], self.max_pairs_per_cluster, replace=False)]
			S_list.append(S)
			D_list.append(D)
		return S_list, D_list

	def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

		device = feats.device
		if feats.dim() != 2:
			feats = feats.view(feats.shape[0], -1)
		N, D = feats.shape
		if N <= 1:
			return feats.new_zeros(())


		self._ensure_params(D, device)


		X_np = feats.detach().cpu().numpy()  # (N, D)
		y_np = labels.detach().cpu().view(-1).numpy()
		bin_labels_all = self._label_bin(y_np)  # (N,)


		k_target = int(self.max_clusters)
		labels_km, C_km = self._kmeans_cpu(X_np, k=k_target, random_state=self.random_state)

		R_cells_km = self._nearest_receptive(X_np, C_km) if C_km.shape[0] > 0 else []


		valid_centers: List[np.ndarray] = []
		valid_R: List[np.ndarray] = []
		present_bins: List[int] = []
		old_to_new: Dict[int, int] = {}
		for old_idx, inds in enumerate(R_cells_km):
			if inds.size == 0:
				continue
			valid_centers.append(C_km[old_idx])
			valid_R.append(inds.copy())
			bins_local = bin_labels_all[inds]
			vals, cnts = np.unique(bins_local, return_counts=True)
			present_bins.append(int(vals[cnts.argmax()]))
			old_to_new[old_idx] = len(valid_centers) - 1

		if not valid_centers:
			return feats.new_zeros(())

		C_use = np.stack(valid_centers, axis=0)
		R_cells_use = valid_R
		labels_cat = np.array([old_to_new[int(lbl)] for lbl in labels_km], dtype=np.int64)
		K = C_use.shape[0]

		self._cached_assignments = [rc.copy() for rc in R_cells_use]
		self._cached_features = feats.detach().cpu().clone()
		self._cached_labels_per_sample = labels_cat.copy()


		S_list, D_list = self._build_pairs(R_cells_use, bin_labels_all, K)

		# pdb.set_trace()


		X = feats.t()  # (D, N)
		C = torch.as_tensor(C_use, dtype=feats.dtype, device=device).t()  # (D, K)


		M_0 = self.M_0  # (D, D)
		Delta_M_list = self.Delta_M[:K]  # (K, D, D)
		M_list = M_0.unsqueeze(0) + Delta_M_list  # (K, D, D)

		# pdb.set_trace()

		I = torch.eye(D, dtype=feats.dtype, device=device)
		total = feats.new_zeros(())


		labels_cat_t = torch.as_tensor(labels_cat, device=device)

		for i in range(K):
			Mi = M_list[i] + self.beta * I  # (D,D)

			S_i = S_list[i]
			D_i = D_list[i]
			s_sum = None
			d_sum = None
			if S_i.size > 0:
				p = torch.as_tensor(S_i[:, 0], device=device, dtype=torch.long)
				q = torch.as_tensor(S_i[:, 1], device=device, dtype=torch.long)
				diffs = X[:, p] - X[:, q]  # (D, P)
				s_sum = (diffs * (Mi @ diffs)).sum()
			if D_i.size > 0:
				p = torch.as_tensor(D_i[:, 0], device=device, dtype=torch.long)
				q = torch.as_tensor(D_i[:, 1], device=device, dtype=torch.long)
				diffs = X[:, p] - X[:, q]  # (D, Q)
				d_sum = (diffs * (Mi @ diffs)).sum()

			if (s_sum is not None) or (d_sum is not None):
				if s_sum is None:
					s_sum = torch.zeros((), dtype=feats.dtype, device=device)
				if d_sum is None:
					d_sum = torch.zeros((), dtype=feats.dtype, device=device)
				total = total + torch.relu(s_sum - d_sum + self.xi)


			ci = C[:, i]
			k_idx_i = torch.nonzero(labels_cat_t == i, as_tuple=False).view(-1)
			for j in range(K):
				if j == i:
					continue
				Mj = M_list[j] + self.beta * I
				cj = C[:, j]
				rho_ij = self.alpha_ * torch.norm(ci - cj).pow(2)


				if k_idx_i.numel() > 0:
					xi = X[:, k_idx_i]
					pos = xi - ci.view(-1, 1)
					neg = xi - cj.view(-1, 1)
					d_pos = (pos * (Mi @ pos)).sum(dim=0)
					d_neg = (neg * (Mj @ neg)).sum(dim=0)
					total = total + self.v1 * torch.relu(-(d_neg - d_pos - rho_ij)).sum()




		frobenius_penalty = self.v2 * torch.norm(M_0, p='fro').pow(2)
		total = total + frobenius_penalty


		loss = total / max(1, N)
		

		self._cached_centers = C_use  
		self._cached_K = K
		self._cached_centers_bins = present_bins
		self._M_0 = M_0
		
		return loss
	
	def get_current_clusters_info(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list, float]:

		if not hasattr(self, '_cached_centers') or not hasattr(self, '_cached_K'):

			dummy_dim = self._feat_dim if self._feat_dim else 1
			return torch.empty(0, dummy_dim, device=device), \
				   torch.empty(0, dummy_dim, dummy_dim, device=device), \
				   torch.empty(0, dummy_dim, dummy_dim, device=device), \
				   0, [], 0.0
		
		K = self._cached_K
		if K == 0:
			return torch.empty(0, self._feat_dim, device=device), \
				   torch.empty(0, self._feat_dim, self._feat_dim, device=device), \
				   torch.empty(0, self._feat_dim, self._feat_dim, device=device), \
				   0, [], 0.0
		

		centers = torch.as_tensor(self._cached_centers, dtype=torch.float32, device=device)  # (K, D)
		

		M_0 = self._M_0.to(device)  # (D, D)
		Delta_M_list = self.Delta_M[:K].to(device)  # (K, D, D)
		bins = self._cached_centers_bins
		
		return centers, M_0, Delta_M_list, K, bins, self.beta
