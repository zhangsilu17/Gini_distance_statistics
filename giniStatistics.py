from sklearn import preprocessing
from scipy.spatial.distance import pdist,squareform
import random
import numpy as np

def GMD(x,sorted=False): 
	# nlog(n) implementation of GMD for 1-dimensional x
	if not sorted:
		x = np.sort(x)
	Delta = 0
	n = len(x)
	for i in range(n):
		Delta = Delta + (2*i-n+1)*x[i]
	Delta = Delta*2/(n*(n-1))
	return Delta

def giniStat_joint(X,y,use_sample='auto',sigma2=10):
	# Measures Gini statistics between X and y
	# Parameters:
	# 	X: array_like, shape(n_samples,n_features)
	#		Feature matrix
	#	y: array_like, shape(n_samples,), treated as discrete values
	#		Target vector
	# 	use_sample: {int, 'auto',None}, default 'auto'
	# 		If int, it is the number of samples used to calculate gCov/gCor. If 'auto', use 5000 random samples if n_samples>5000.
	# 		If None, all data will be used
	#	sigma2: {float,None}, default 10
	#		The kernal parameter for generalized Gini distance statistics. If None, the non-kernal version will be calculated
	# Retures:
	# v: float
	# 	Estimated Gini distance corvariance
	# r: float
	# 	Estimated Gini distance correlation
	if use_sample == 'auto' and len(y)>5000:
		sample = random.sample(range(len(y)),5000)
		X = X[sample]
		y = y[sample]
	elif type(use_sample) is int:
		sample = random.sample(range(len(y)),use_sample)
		X = X[sample]
		y = y[sample] 
	y_nuiq = np.unique(y)
	K = len(y_nuiq)
	# remove classes with only one sample
	for c in y_nuiq:
		if np.sum(y==c)==1:
			keep = (y!=c)
			X = X[keep,:]
			y = y[keep]
	# encode y
	le = preprocessing.LabelEncoder()
	le.fit(y)
	y=le.transform(y)
	K = len(np.unique(y))
	n = X.shape[0]
	# standardize X
	X = preprocessing.scale(X)
	X = np.nan_to_num(X)
	if len(X.shape)==1 and sigma2 is None: # use O(nlog(n)) implementation
		idx = np.argsort(X)
		x = X[idx]
		y = y[idx]
		Delta = GMD(x,sorted=True)
		if Delta==0:
			return 0,0
		v = Delta
		for k in range(K):
			Delta_k = GMD(x[y==k],sorted=True)
			p_k = sum(y==k)/n
			v = v-p_k*Delta_k
		r = v/Delta
		return v,r
	else:	# O(n^2) implementation	
		if len(X.shape)==1:
			X=X.reshape((-1,1))
		Y = np.zeros((n,K))
		for i in range(n):
			Y[i,y[i]] = 1/np.sqrt(2)
		if sigma2 is not None:
			D = squareform(pdist(X, 'sqeuclidean'))
			D = np.sqrt(1-np.exp(D/(-sigma2)))
		else:
			D = squareform(pdist(X, 'euclidean'))
		Delta = np.sum(D)/(n*(n-1))
		if Delta==0:
			return 0,0
		v = Delta
		# eps = 10**(-6)
		for k in range(K):
			idx = np.where(Y[:,k]>0)[0]
			Delta_k = np.sum(D[idx,:][:,idx])/(len(idx)*(len(idx)-1))
			v = v - len(idx)/n*Delta_k
		r = v/Delta
		return v,r

def giniStat_marignal(X,y,use_sample='auto',sigma2=10):
	# Measures Gini statistic between each feature in X and y
	# Parameters:
	# 	X: array_like, shape(n_samples,n_features)
	#		Feature matrix
	#	y: array_like, shape(n_samples,), treated as discrete values
	#		Target vector
	# 	use_sample: {int, 'auto',None}, default 'auto'
	# 		If int, it is the number of samples used to calculate gCov/gCor. If 'auto', use 5000 random samples if n_samples>5000.
	# 		If None, all data will be used
	#	sigma2: {float,None}, default 10
	#		The kernal parameter for generalized Gini distance statistics. If None, the non-kernal version will be calculated
	# Retures:
	# v: array_like, shape(n_features,)
	# 	Estimated Gini distance corvariance between each feature and y
	# r: array_like, shape(n_features,)
	# 	Estimated Gini distance correlation between each feature and y
	d = X.shape[1]
	v = np.zeros((d,))
	r = np.zeros((d,))
	for i in range(d):
		x = X[:,i]
		v_i,r_i = giniStat_joint(x,y,use_sample=use_sample,sigma2=sigma2)
		v[i] = v_i
		r[i] = r_i
	return v,r