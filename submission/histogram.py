import pandas as pd
from dataloader import Dataloader
from viz import plot_grouped_histogram
import sys
import os

def transform_string(s):
	return "" + s.lower().replace(" ", "_")
def ensure_folder_exists(folder_path):
	os.makedirs(folder_path, exist_ok=True)
	return folder_path

if __name__ == "__main__":
	if len(sys.argv) == 2 and sys.argv[1][-4:] == ".csv":
		filename = sys.argv[1]
		expl_columns = ['Hogwarts House', 'Best Hand', 'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
						'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 
						'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
		target_col = 'Hogwarts House'

		datas = Dataloader(filename=filename, 
					expl_columns=expl_columns, 
					target_col=target_col,
					isIndex=True)
		
		for i, variable in enumerate(expl_columns):
			var = variable
			if var != target_col:
				plot = plot_grouped_histogram(datas._dataframe, var, target_col, f"Distribution of {var} score per house")
				to_file = transform_string(var)
				plot.write_html(ensure_folder_exists('./viz/') + f"histogram_{to_file}.html")
		# HINT CARE OF MAGICAL CREATURE AND ARITHMANCY
	else:
		print("Error: arg must be:\n>>> python path/to/histogram.py dataset.csv")
