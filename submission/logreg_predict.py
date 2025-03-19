from dataloader import Dataloader
from model import LogisticRegression
import numpy as np
from viz import plot_losses, plot_accuracy
import os
import pandas as pd
import json
import sys

import json
import numpy as np

def ensure_folder_exists(folder_path):
	os.makedirs(folder_path, exist_ok=True)
	return folder_path

def load_model_parameters(file_path):
	with open(file_path, "r") as f:
		model_data = json.load(f)
	
	weights = np.array(model_data["weights"])
	bias = np.array(model_data["bias"])
	norm_params = np.array(model_data["normalization_parameters"])
	class_name = list(model_data["class_order"])
	return weights, bias, class_name, norm_params



if len(sys.argv) == 3 and sys.argv[1][-4:] == ".csv" and sys.argv[2][-5:] == ".json":
	filename = sys.argv[1]
	weights_path = sys.argv[2]

	weights, bias, class_name, norm_params = load_model_parameters(file_path=weights_path)
	mean, std = norm_params[0], norm_params[1]
	expl_columns = ['Best Hand', 'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
				'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 
				'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
	try:
		datas = Dataloader(filename=filename,
					expl_columns=expl_columns,
					target_col=None,
					isIndex=True)
	except Exception as e:
		print('ERROR:', e)
		sys.exit(0)

	model = LogisticRegression(len(datas._dataframe.columns), len(class_name))
	X = np.array(datas._dataframe)
	X = (X - mean) / std
	model.weights = weights
	model.bias = bias
	preds = model.predict(X, class_name)
	output = {"Hogwarts House": [class_name[np.argmax(vals)] for vals in preds] }
	output = pd.DataFrame(output)
	
	output.index.name = "Index"
	output.to_csv(ensure_folder_exists('./prediction/') + 'houses.csv')
else:
	print("Error: arg must be:\n>>> python path/to/logreg_predict.py dataset_test.csv path/to/weights.json")