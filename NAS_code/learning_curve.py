# Load libraries
import scipy.io 
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from args import parser
args = parser.parse_args()


def plot_curve(file, Multi_running):
	
	mat = scipy.io.loadmat(file)
	train_scores = list(itertools.chain.from_iterable(mat['train_acc']))
	test_scores = list(itertools.chain.from_iterable(mat['test_acc']))

	train_sizes = np.linspace(0.0, len(test_scores)-1, len(test_scores))

	if Multi_running:	## =========== Multi running ======================
		train_mean = np.mean(train_scores, axis=1)
		train_std = np.std(train_scores, axis=1)
		# Create means and standard deviations of test set scores
		test_mean = np.mean(test_scores, axis=1)
		test_std = np.std(test_scores, axis=1)

		# Draw lines
		plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
		plt.plot(train_sizes, test_mean, color="#111111", label="Test score")

		# Draw bands
		plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
		plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

		# Create plot
		plt.title("Learning Curve")
		plt.xlabel("Epoch"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
		plt.tight_layout()
		plt.show()


	else: #===== Single running =================
		plt.plot(train_sizes, train_scores,  color="#111111",  label="Training score" )
		plt.plot(train_sizes, test_scores, '--', color="#111111", label="Test score")

		# Create plot
		plt.title("Learning Curve", {'size':20})
		plt.xlabel("Epoch",{'size':20}), plt.ylabel("Accuracy Score", {'size':20}), plt.legend(loc="best", prop ={'size':18})
		plt.tight_layout()
		plt.show()


plot_curve('../results/test_acc0.5653_param5.0MB.mat', Multi_running = False)