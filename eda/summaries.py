import numpy as np
import pandas as pd
import statistics

class statoperation:
	@staticmethod
	def mean_lla(lst):
		return np.sum(lst) / len(lst)

	@staticmethod
	def variance(lst, mean=None):
		av = mean if mean is not None else statoperation.mean_lla(lst)
		residual = (lst - av)**2
		return np.sum(residual) / len(lst)

	@staticmethod
	def std(variance):
		return np.sqrt(variance)

	@staticmethod
	def q1(data):
		data.sort()
		q1 = statoperation.mediane(data[:len(data)//2])
		return q1
	
	@staticmethod
	def mediane(data):
		data.sort()
		if len(data) % 2 == 0: # even
			return (data[(len(data) // 2) - 1] + data[(len(data) // 2)]) / 2
		else:

			return data[(len(data)) // 2]

	@staticmethod
	def q2(data):
		data.sort()
		return statoperation.mediane(data)

	@staticmethod
	def q3(data):
		data.sort()
		q3 = statoperation.mediane(data[(len(data)+1)//2:])
		return q3
	
	@staticmethod
	def min(data):
		data.sort()
		return data[0]

	@staticmethod
	def max(data):
		data.sort()
		return data[-1]

# Lets sum-up our continuous variables
def continuous_variable_summary(df, col_name):
	mean = []
	var = []
	std = []
	min_vals = []
	max_vals = []
	med = []
	q1 = []
	q3 = []

	for col in col_name:
		np_array = np.array(df[col])
		mean.append(statoperation.mean_lla(np_array))
		variance = statoperation.variance(np_array)
		var.append(variance)
		std.append(statoperation.std(variance))
		min_vals.append(statoperation.min(np_array))
		max_vals.append(statoperation.max(np_array))
		med.append(statoperation.q2(np_array))
		q1.append(statoperation.q1(np_array))
		q3.append(statoperation.q3(np_array))

	summary = pd.DataFrame({
		"mean": mean,
		"variance": var,
		"std_dev": std,
		"min": min_vals,
		"max": max_vals,
		"median": med,
		"q1": q1,
		"q3": q3

	}, index=col_name)

	return summary

def check_null_ish_values(df):
	null_counts = df.isnull().sum()  # NaN or None values
	empty_string_counts = (df == "").sum()  # Empty strings
	custom_nulls = ["unknown", "N/A", "na"]  # Add your specific "null-ish" values here
	custom_counts = {val: (df == val).sum() for val in custom_nulls}

	# Combine results into a DataFrame
	summary = pd.DataFrame({
		"Nulls": null_counts,
		"Empty Strings": empty_string_counts,
		**{f"Custom ({val})": count for val, count in custom_counts.items()}
	})
	
	return summary