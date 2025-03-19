import copy
import numpy as np
from sklearn.metrics import accuracy_score
from summaries import statoperation as stats
from summaries import custom_mean


class LogisticRegression():
	def __init__(self, num_expl_var, num_class):
		self.losses = []
		self.train_accuracies = []
		self.weights = np.zeros((num_class, num_expl_var))
		self.bias = np.zeros(num_class)


	def fit(self, x, y, epochs, class_name=None):
		x = x.copy()
		y = y.copy()
		for i in range(epochs):
			z = np.matmul(x, self.weights.T) + self.bias
			pred = self._sigmoid(z)
			loss = self.compute_loss(y, pred)
			error_w, error_b = self.compute_gradients(x, y, pred)
			self.update_model_parameters(error_w, error_b)

			pred_to_class = np.vectorize(self._sanitize_pred)(pred)
			accuracies = {}
			losse_dict = {}
			for idx in range(y.shape[1]):
				key = str(idx) if class_name is None else class_name[idx]
				accuracies.update({ key: accuracy_score(y[:, idx], pred_to_class[:, idx]) })
				losse_dict.update({ key: loss[idx] })
			self.train_accuracies.append(accuracies)
			self.losses.append(losse_dict)
			self._log(i + 1, epochs, modulo=20)

	def predict(self, x, class_name=None):
		x = x.copy()
		z = np.matmul(x, self.weights.T) + self.bias
		pred = self._sigmoid(z)
		return pred

	def compute_loss(self, y_true, y_pred):
		# binary cross entropy
		y_zero_loss = y_true * np.log(y_pred + 1e-9)
		y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
		return -custom_mean(y_zero_loss + y_one_loss, axis=0)

	def compute_gradients(self, x, y_true, y_pred):
		# derivative of binary cross entropy
		difference =  y_pred - y_true
		gradient_b = custom_mean(difference, axis=0)
		gradients_w = np.matmul(x.T, difference) * 1/len(y_true) # TODO maybe add 1 / n
		# gradients_w = np.array([np.mean(grad) for grad in gradients_w])
		return gradients_w, gradient_b

	def update_model_parameters(self, error_w, error_b):
		self.weights = self.weights - 0.1 * error_w.T
		self.bias = self.bias - 0.1 * error_b

	def _sigmoid(self, x):
		ret = np.array([self._sigmoid_function(value) for value in x])
		return ret

	def _sigmoid_function(self, x):
		def exp_trick(x):
			if x >= 0:
				z = np.exp(-x)
				return 1 / (1 + z)
			else:
				z = np.exp(x)
				return z / (1 + z)
		if isinstance(x, np.ndarray):
			return [exp_trick(val) for val in x]
		else:
			return exp_trick(x)

	def _transform_x(self, x):
		x = copy.deepcopy(x)
		return x.values

	def _transform_y(self, y):
		y = copy.deepcopy(y)
		return y.values.reshape(y.shape[0], 1)
	
	def _sanitize_pred(self, x):
		if x > 0.5: 
			return 1
		return 0

	def _log(self, i, epochs, modulo):
		if i % modulo == 0:
			print(f'LOG: epoch {i}/{epochs}\nLosses:')
			for key, val in self.losses[-1].items():
				print(f'{key} : {val}')
	

