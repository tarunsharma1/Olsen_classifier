import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from numpy import linalg as LA
import math
import pickle

global_weights = pickle.load(open('global_weights_subsample_negative.p','rb'))
global_data_X = pickle.load(open('global_data_X_subsample_negative.p','rb'))
global_data_Y = pickle.load(open('global_data_Y_subsample_negative.p','rb'))
noise_distributions = pickle.load(open('noise_distributions.p','rb'))
PN_responses = pickle.load(open('PN_responses_classifier.p','rb'))

print ('done reading weights and data')

nOdors = 183

##### before the testing ...we should use the training data to find the threshold..i.e ensure that false positives = false negatives ###
for threshold in np.linspace(40,60,100):
	total_fp = 0
	total_fn = 0
	total_tp = 0
	total_tn = 0
	# for each perceptron
	for p in range(0,nOdors):
		dataX = global_data_X[p]
		dataY = global_data_Y[p]
		weights = global_weights[p,:]

		tp = 0
		fp = 0
		tn = 0
		fn = 0
		
		for idx,entry in enumerate(dataX):
			label = dataY[idx]
			dot_product = np.dot(entry, weights) 
					
			if dot_product >= threshold and label==1:
				tp+=1
				
			elif dot_product >= threshold and label==0:
				fp+=1
				
			elif dot_product <= threshold and label==1:
				fn+=1
			
			elif dot_product <= threshold and label==0:
				tn+=1
		
		total_tp += tp/100.0
		#total_fp += fp/15700.0
		total_fp += fp/100.0
		
		total_fn += fn/100.0
		#total_tn += tn/15700.0
		total_tn += tn/100.0

	print ('threshold:, total fp:,total fn:',threshold,total_fp/nOdors,total_fn/nOdors)
	print ('total_tp rate:, total_tn rate:',total_tp/nOdors, total_tn/nOdors)



# I looked at the values above and manually picked the threshold which has an equal false positive and false negative rate
threshold = 52.92


##### testing for this perceptron using the set threshold
print ('##### TESTING #####')

total_fp_test = 0
total_fn_test = 0
total_tp_test = 0
total_tn_test = 0



for p in range(0,nOdors):
	tp_test = 0
	fp_test = 0
	tn_test = 0
	fn_test = 0
	responses = PN_responses[:,p]
	weights = global_weights[p,:]
	for iterations in range(0,50):
		# randomly draw indices (between 0 and 5000) for each of the 23 responses and add them to the responses
		noise = np.zeros((23))
		for i in range(0,23):
			noise[i] = noise_distributions[p,i,np.random.randint(5000)]

		##### this is positive example for training ########
		noisy_responses = responses + noise
		dot_product = np.dot(noisy_responses, weights) 
		if dot_product >= threshold:
			tp_test+=1
		else:
			fn_test+=1	
		

		## negatives for testing (all other odors)
		for random_odor_negative in range(0,nOdors):
			if random_odor_negative == p:
				continue
			responses_negative = PN_responses[:,random_odor_negative]
			## add noise to the negative sample
			
			noise_negative = np.zeros((23))
			for i in range(0,23):
				noise_negative[i] = noise_distributions[random_odor_negative,i,np.random.randint(5000)]
			responses_negative = responses_negative + noise_negative
			dot_product = np.dot(responses_negative, weights) 
			if dot_product >= threshold:
				fp_test+=1
			else:
				tn_test+=1



	total_tn_test += tn_test/(50.0*(nOdors-1))
	total_fp_test += fp_test/(50.0*(nOdors-1))
	total_fn_test += fn_test/50.0
	total_tp_test += tp_test/50.0


print ('threshold:, total fp:,total fn:',threshold,total_fp/nOdors,total_fn/nOdors)
print ('total_tp rate:, total_tn rate:',total_tp/nOdors, total_tn/nOdors)

# in the paper they repeated the whole process of training and testing for 500 times and took the average. I did not do this. 
