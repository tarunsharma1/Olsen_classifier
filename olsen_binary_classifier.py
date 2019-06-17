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


mat = scipy.io.loadmat('../Original_HC_no_fruit_conc_with_fictive1.mat')
#mat = scipy.io.loadmat('../KCsim_HC_and_fake_1.mat')

# check for both ORN and KC that spontaneous activity has been subtracted or not
ORN = mat['ORN']
ORN = ORN['rates'][0][0] # 51,nOdors



# first get PN responses (average firing rate)...for now assuming ORN is the PN input (ORN.shape = 51,nOdors)
#PN_responses = ORN # 51,nOdors
PN_responses = mat['PN']['rates_RW'][0][0]

print (PN_responses.shape)
#import ipdb;ipdb.set_trace()

# select only the non zero ones i.e the 23 which have responses
nonzero_PN_indices = np.where(np.array([np.sum(PN_responses[i,:]) for i in range(0,51)])!=0)[0]
PN_responses = PN_responses[nonzero_PN_indices,:] # 23,nOdors


nOdors = 183


# based on values in PN responses and formula in book, calculate SD and then make gaussian distribution to draw noise value from

# one perceptron per odor... start with random weights ...look at equation for perceptron weight update rule (make sure that weights are non negative)
# threshold c is same for one entire set of odor responses. This is adjusted during training so that false hits rate = rate of false misses
# for training data :
# for each perceptron -


threshold = 50 # I start off with a random value and then select the threshold value in the testing part of the code where I ensure that 
# rate of false positives = rate of false negatives



print ('computing noise distributions')
noise_distributions = np.zeros((nOdors,23,5000))
for k in range(0,nOdors):
	responses = PN_responses[:,k]
	for i in range(0,23):
		mean = responses[i]
		sd = abs(9.5 - 7.2*math.exp(-1*mean/76.0))
		# randomly creating 5000 points
		noise_distributions[k,i,:] = np.random.normal(mean,sd,5000)
print ('noise computation done')



global_weights = np.zeros((nOdors,23))
global_data_X = []
global_data_Y = []

for p in range(0,nOdors):
	
	#	randomly initialize weights
	weights = np.random.randint(10,size=23)/100.0
	responses = PN_responses[:,p] # 23
	#	for iterations in range (0,100):
	#		for each value (23) in the response to that odor, use the value to get SD (noise) independently and add it to the value
	#		update the weights for that perceptron based on equation ..positive training example is the original response+noise. 
	#		Im not sure of negative example...either 100 randomly selected (i.e not this odor) responses be used or all the non odor responses
	#		be used...i think even for negative examples we should add noise.	
	
	
	# add whatever training data to these in order to use to find threshold after training
	dataX = []
	dataY = []

	for iterations in range(0,100):
		#print ('##iteration: ',iterations)
		# randomly draw indices (between 0 and 5000) for each of the 23 responses and add them to the responses
		noise = np.zeros((23))
		for i in range(0,23):
			noise[i] = noise_distributions[p,i,np.random.randint(5000)]

		##### this is positive example for training ########
		noisy_responses = responses + noise
		label = 1

		dataX.append(noisy_responses)
		dataY.append(label)

		dot_product = np.dot(noisy_responses, weights) 

		if dot_product >= threshold:
			prediction = 1
		else:
			prediction = 0
		# l1 norm...can try later with l2 norm
		temp = weights - (prediction - label)*noisy_responses/LA.norm(noisy_responses,1)
		# check if update will still keep weights non negative..those weights that will become negative should not be updated
		negative_weight_indices = np.where((temp<0))[0]
		for z in negative_weight_indices:
			# restore those to previous value
			temp[z] = weights[z]
		weights = temp

	

		##### negative examples for training ##########
		#n_updates = 0
		# for this im not 100% sure this is what they did...i will subsample and select one of the other responses apart from p and also add noise to them
		# this can be changed to have all other odors as negative. I tried that and did not get the same values for tp as in the paper.
		
		#for random_odor_negative in range(0,nOdors):
		random_odor_negative = np.random.randint(nOdors)
		while random_odor_negative == p:
			random_odor_negative = np.random.randint(nOdors)

		#if random_odor_negative == p:
		#	continue
		responses_negative = PN_responses[:,random_odor_negative]
		## add noise to the negative sample
		
		noise_negative = np.zeros((23))
		for i in range(0,23):
			noise_negative[i] = noise_distributions[random_odor_negative,i,np.random.randint(5000)]
		responses_negative = responses_negative + noise_negative
		label = 0

		dataX.append(responses_negative)
		dataY.append(label)


		dot_product = np.dot(responses_negative, weights) #+ weights[-1]
		if dot_product >= threshold:
			prediction = 1
			
		else:
			prediction = 0
		# l1 norm...can try later with l2 norm
		temp = weights - (prediction - label)*responses_negative/LA.norm(responses_negative,1)
		# check if update will still keep weights non negative..those weights that will become negative should not be updated
		negative_weight_indices = np.where((temp<0))[0]
		for z in negative_weight_indices:
			# restore those to previous value
			temp[z] = weights[z]
		weights = temp
	
	global_weights[p,:] = weights
	global_data_X.append(dataX)
	global_data_Y.append(dataY)

print ('training for all perceptrons done')
pickle.dump(global_weights,open('global_weights_subsample_negative.p','wb'))
pickle.dump(global_data_X,open('global_data_X_subsample_negative.p','wb'))
pickle.dump(global_data_Y,open('global_data_Y_subsample_negative.p','wb'))
pickle.dump(noise_distributions,open('noise_distributions.p','wb'))
pickle.dump(PN_responses,open('PN_responses_classifier.p','wb'))