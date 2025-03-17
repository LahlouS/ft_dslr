from viz import plot_scatter, plot_correlation_heatmap
from dataloader import Dataloader
import sys
import numpy as np

def transform_string(s):
	return "" + s.lower().replace(" ", "_")


def zscore_correlation_matrix(data):
	# Compute the mean and standard deviation for each column
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0, ddof=1)  # Use ddof=1 for sample standard deviation
	
	# Normalize using z-score
	normalized_data = (data - mean) / std
	
	# Compute the correlation matrix
	correlation_matrix = np.corrcoef(normalized_data, rowvar=False)
	
	return correlation_matrix

import numpy as np

def top_k_activations(corr_matrix, k):
	"""
	Finds the top-k unique activation values with the highest absolute correlations 
	(both positive and negative) and their indices from a symmetric correlation matrix.
	
	Parameters:
		corr_matrix (np.ndarray): The NxN correlation matrix (assumed to be symmetric).
		k (int): Number of top activations to return.
	
	Returns:
		List of tuples: [(value, (row_idx, col_idx)), ...]
	"""
	assert corr_matrix.shape[0] == corr_matrix.shape[1], "Matrix must be square"
	
	# Get upper triangle indices (excluding diagonal)
	triu_indices = np.triu_indices_from(corr_matrix, k=1)
	
	# Extract unique correlation values
	unique_values = corr_matrix[triu_indices]
	
	# Sort by absolute value but keep the original sign
	top_k_indices = np.argsort(np.abs(unique_values))[-k:][::-1]
	
	# Map back to (row, col) indices
	top_k_values = [(unique_values[i], (triu_indices[0][i], triu_indices[1][i])) for i in top_k_indices]
	
	return top_k_values
	


if __name__ == "__main__":
	if len(sys.argv) == 2 and sys.argv[1][-4:] == ".csv":
		filename = sys.argv[1]
		expl_columns = ['Best Hand', 'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
						'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 
						'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
		target_col = 'Hogwarts House'

		datas = Dataloader(filename=filename, 
					expl_columns=expl_columns, 
					target_col=target_col,
					isIndex=True)
		
		X = np.array(datas._dataframe)
		corr_mat = zscore_correlation_matrix(X)
		fig_heatmap = plot_correlation_heatmap(corr_mat, variable_names=datas._dataframe.columns)
		fig_heatmap.write_html("viz/correlation_heatmap.html")
		top_k = top_k_activations(corr_mat, k=5)

		col = datas._dataframe.columns
		for stuff in top_k:
			print(f'({col[stuff[1][0]]} / {col[stuff[1][1]]}) -> activation:', stuff[0])
			varx = col[stuff[1][0]]
			vary = col[stuff[1][1]]

			fig_scatter = plot_scatter(x=datas._dataframe[varx], y=datas._dataframe[vary], group=datas.classes_original, axis_name=(varx, vary))
			to_filex, to_filey = transform_string(varx), transform_string(vary)
			fig_scatter.write_html(f'viz/scatter_{to_filex}_{to_filey}.html')
	else:
		print("Error: arg must be:\n>>> python path/to/describe.py dataset.csv")


