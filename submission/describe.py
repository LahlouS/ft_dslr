from summaries import continuous_variable_summary

from dataloader import Dataloader
import pandas as pd
import sys

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
		df = datas._get_float_cols(datas._dataframe)
		print(continuous_variable_summary(df, df.columns))
	else:
		print("Error: arg must be:\n>>> python path/to/describe.py dataset.csv")