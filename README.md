# Olsen_classifier

This is a project related to the drosophila (fruit fly) olfactory system. This is an implementation of a part of 
the following paper - 

Citation : 
 Olsen SR, Bhandawat V, Wilson RI. Divisive normalization in olfactory population codes. Neuron. 2010;66:287–299.
 
https://www.cell.com/action/showPdf?pii=S0896-6273%2810%2900249-7

This is to access the odor discriminability of the fly based on the PN (projection neuron) responses in the brain of the 
fly in response to different odors. A perceptron is trained for each odor where the positive training examples are the
responses to that odor plus noise (statistical gaussian noise based on formula in paper) and negative examples are responses
of other odors. The same training data is then saved and is used after training to pick a common threshold for all the 
classifiers. This is done by picking a threshold where the overall rate of false positives is equal to the overall rate
of false negatives. 

Once this is done the model is tested by generating new responses with noise and the true positive rate is noted.

To me it was not clear in the original paper whether they used all the other odor responses as negatives during training
or they just subsampled and picked a random odor. I tried the first approach and was not able to get the same result but the
second approach gave me about the same value. 

I was able to get the mean true positive rate using the ORN responses ('untransformed' in the paper) to 62%. This is in close
agreement with what is reported in the paper (65%). For the PN responses, the model I was using used a different ORN to PN
transformation than what is used in the paper and as a result I got different values.

For further details contact me at tarunsharma.pes@gmail.com
