from dataloader import Dataloader
from model import LogisticRegression
import numpy as np
from viz import plot_losses, plot_accuracy
import os
import pandas as pd
import json
import sys
from summaries import custom_mean, custom_std

def ensure_folder_exists(folder_path):
	os.makedirs(folder_path, exist_ok=True)
	return folder_path

def z_normalize(data):
	mean = custom_mean(data, axis=0)
	std = custom_std(data, axis=0)  # Use ddof=1 for sample standard deviation
	normalized_data = (data - mean) / std
	return normalized_data, mean, std

def save_model_parameters(weights, bias, class_name, norm_params, file_path):
	model_data = {
		"weights": weights.tolist(),
		"bias": bias.tolist(),
		"class_order": class_name,
		"normalization_parameters": norm_params.tolist()
	}
	
	with open(file_path, "w") as f:
		json.dump(model_data, f, indent=4)

if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1][-4:] == ".csv":
		filename = sys.argv[1]
		basepath = ''
		if len(sys.argv) > 2 and sys.argv[2][-1] == '/' :
			basepath = ensure_folder_exists(sys.argv[2])
		elif len(sys.argv) == 2:
			pass
		elif not sys.argv[2][-1] == '/':
			print("error: basepath arg must end by \'/\'")
			sys.exit(0)
		expl_columns = ['Best Hand', 'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
						'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 
						'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
		target_col = 'Hogwarts House'

		datas = Dataloader(filename=filename, 
					expl_columns=expl_columns, 
					target_col=target_col,
					isIndex=True)

		model = LogisticRegression(len(datas._dataframe.columns), len(datas._classes))

		X = np.array(datas._dataframe)
		X, means, stds = z_normalize(X)
		y = np.array(datas._target_matrix)
		class_name = datas._classes

		model.fit(X, y, 200, class_name)

		loss_plot = plot_losses(model.losses)
		acc_plot = plot_accuracy(model.train_accuracies)

		loss_plot.write_html(ensure_folder_exists('./viz/') + "model_losses.html")
		acc_plot.write_html(ensure_folder_exists('./viz/') + "accuracy_plot.html")

		norm_param = np.vstack((means, stds))
		
		save_model_parameters(model.weights, model.bias, class_name, norm_param, ensure_folder_exists('./weights/') + "weights.json")
	else:
		print("Error: arg must be:\n>>> python path/to/logreg_train.py.py dataset.csv [opional outfile_basepath]")
