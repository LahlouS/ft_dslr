import numpy as np
import pandas as pd
from summaries import statoperation as stats

class Dataloader(object):
	def __init__(self, filename, expl_columns, target_col=None, isIndex=False):
		self.filename = filename
		self.expl_columns = expl_columns
		try:
			dataframe = pd.read_csv(filename, index_col=None if isIndex == False else 0)
			if target_col is not None:
				self.classes_original = dataframe[target_col]
			self._dataframe = self._clean_data(dataframe)[self.expl_columns].__deepcopy__()
			
			binary_classes = self._get_binary_classes(self._dataframe)
			for col in binary_classes:
				self._dataframe[col] = self._handle_binary_class(self._dataframe[col])

			if target_col is not None:
				self._classes = list(dataframe[target_col].unique())
				self._target_matrix = pd.DataFrame()
				for cl in self._classes:
					self._target_matrix[cl] = self._class_to_target(dataframe[target_col], cl)

		except Exception as e:
			raise Exception(f"{type(e)} Error occur processing the data:{e}")

	def __getitem__(self, idx):
		if idx < self.__len__():
			ret = {}
			for col in self._classes:
				ret.update({col: self._target_matrix.loc[idx, col]})
			for col in self._dataframe.columns:
				ret.update({col : self._dataframe.loc[idx, col]})
			return ret
		return None

	def _clean_data(self, df):
		"""
			this function replace every NaN or null-ish values 
			in the data by the mean of its corresponding variable
		"""
		for col in self._get_float_cols(df):
			df[col] = self._replace_nan(df[col])
		return df

	def _handle_binary_class(self, binary_list):
		unique_value = binary_list.unique()
		if len(unique_value) != 2:
			raise TypeError("_handle_binary_class: list is not binary")
		def transform(x):
			if x == unique_value[0]:
				return 1
			elif x == unique_value[1]:
				return 0
			else:
				raise Exception("binary class transformation made impossible")
		binary_list = binary_list.apply(transform)
		return binary_list 
	
	def _replace_nan(self, col):
		filled = col.fillna(0)
		av = stats.mean_lla(filled)
		return col.fillna(av)

	def _get_float_cols(self, df):
		return df.select_dtypes(include=['float'])
	
	def _get_binary_classes(self, df):
		lst = []
		for col in df.select_dtypes(include=['object']).columns:
			if len(df[col].unique()) == 2:
				lst.append(col)
		return lst

	def _class_to_target(self, targets, target_name):
		def transform(x):
			if x == target_name:
				return 1
			return 0
		return targets.apply(transform)
	
	def __len__(self):
		return len(self._dataframe)



	