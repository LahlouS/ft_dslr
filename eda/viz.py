import plotly.express as px
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
