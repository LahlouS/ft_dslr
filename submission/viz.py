import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_grouped_histogram(df, variable_col='variable', group_col='group', title="Overlayed Histogram of Variable by Group"):
	"""
	Creates an overlayed histogram comparing the overall distribution of a variable 
	with the distributions of different groups in a dataframe.

	Parameters:
		df (pd.DataFrame): Input DataFrame with at least two columns.
		variable_col (str): Column name for numerical values.
		group_col (str): Column name for categorical groups.
		output_file (str): Output HTML file path.

	Returns:
		None (saves the plot to an HTML file).
	"""
	if variable_col not in df.columns or group_col not in df.columns:
		raise ValueError(f"Columns '{variable_col}' and '{group_col}' must be in the dataframe")

	# Create histogram with overlay
	fig = px.histogram(df, 
						x=variable_col, 
						color=group_col, 
						barmode='overlay',  # Overlay histograms
						opacity=0.6,  # Transparency to see overlap
						marginal='rug',  # Small rug plot for extra info
						title=title)

	return fig


def plot_scatter(x, y, group, axis_name=("x", "y")):
	"""
	Creates a scatter plot using Plotly with different colors for each group.
	
	Parameters:
		x (list): List of x-axis values.
		y (list): List of y-axis values.
		group (list): List of group labels corresponding to each (x, y) point.
	
	Returns:
		plotly.graph_objects.Figure: The generated scatter plot.
	"""
	df = pd.DataFrame({axis_name[0]: x, axis_name[1]: y, 'group': group})
	fig = px.scatter(df, x=axis_name[0], y=axis_name[1], color='group', title='Scatter Plot')
	return fig

def plot_correlation_heatmap(corr_matrix, variable_names=None):
	"""
	Plots a heatmap of the correlation matrix using Plotly.
	
	Parameters:
		corr_matrix (numpy.ndarray): The correlation matrix.
		variable_names (list, optional): List of variable names for axis labels.
	
	Returns:
		plotly.graph_objects.Figure: The generated heatmap.
	"""
	size = corr_matrix.shape[0]  # Ensure square aspect ratio
	
	fig = go.Figure(data=go.Heatmap(
		z=corr_matrix,
		x=variable_names if variable_names is not None else list(range(size)),
		y=variable_names if variable_names is not None else list(range(size)),
		colorscale='Viridis',
		zmin=-1, zmax=1,
		colorbar=dict(title='Correlation')
	))
	
	fig.update_layout(
		title='Correlation Matrix Heatmap',
		xaxis=dict(scaleanchor='y'),  # Ensures square aspect ratio
		yaxis=dict(scaleanchor='x')
	)
	
	return fig


def plot_pairplot(df, group=None):
	"""
	Creates a pair plot matrix using Plotly.
	
	Parameters:
		df (pandas.DataFrame): The input DataFrame with numerical features.
		group (list, optional): List of group labels for color coding.
	
	Returns:
		plotly.graph_objects.Figure: The generated pair plot.
	"""
	if group is not None:
		df = df.copy()
		df['group'] = group
		fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color='group', title='Pair Plot Matrix')
	else:
		fig = px.scatter_matrix(df, dimensions=df.columns, title='Pair Plot Matrix')
	
	return fig


# Example usage
if __name__ == "__main__":
	# Sample Data
	data = {
		"variable": [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10],
		"group": ["A", "A", "B", "B", "A", "B", "A", "A", "B", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
	}
	
	df = pd.DataFrame(data)

	# Generate histogram
	fig = plot_grouped_histogram(df)
	fig.write_html("histogram.html")

def plot_losses(loss_dicts):
	'''
	loss_dicts is of type:
		[{
		"Ravenclaw" : float,
		"Slytherin" : float,
		"Gryffindor" : float,
		"Hufflepuff" : float
		}, ...]
	'''
	epochs = list(range(1, len(loss_dicts) + 1))  # X-axis (number of epochs)
	loss_keys = loss_dicts[0].keys()  # Extract loss names from the first dictionary
	
	# Initialize figure
	fig = go.Figure()
	
	# Add each loss as a separate line
	for key in loss_keys:
		loss_values = [d[key] for d in loss_dicts]  # Extract loss values for each epoch
		fig.add_trace(go.Scatter(x=epochs, y=loss_values, mode='lines', name=key))
	
	# Customize layout
	fig.update_layout(
		title='Loss Values Over Epochs',
		xaxis_title='Epochs',
		yaxis_title='Loss Value',
		template='plotly_dark',
		legend_title='Loss Types'
	)
	return fig

def plot_accuracy(acc_dicts):
	"""
	acc_dict is of type:
		'''
		[{
		"Ravenclaw" : float,
		"Slytherin" : float,
		"Gryffindor" : float,
		"Hufflepuff" : float
		}, ...]
	'''
	"""
	epochs = list(range(1, len(acc_dicts) + 1))  # X-axis (number of epochs)
	acc_keys = acc_dicts[0].keys()  # Extract accuracy keys

	# Initialize figure
	fig = go.Figure()

	# Add each accuracy score as a separate line
	for key in acc_keys:
		acc_values = [d[key] for d in acc_dicts]  # Extract accuracy values for each epoch
		fig.add_trace(go.Scatter(x=epochs, y=acc_values, mode='lines+markers', name=key))

	# Customize layout
	fig.update_layout(
		title='Accuracy Score Over Epochs',
		xaxis_title='Epochs',
		yaxis_title='Accuracy Score',
		template='plotly_dark',
		legend_title='Accuracy Metrics'
	)
	return fig