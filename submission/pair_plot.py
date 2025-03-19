from viz import plot_pairplot
from dataloader import Dataloader
import sys
import numpy as np


def transform_string(s):
	return "" + s.lower().replace(" ", "_")

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
			
			col_to_remove = ["Best Hand", "Astronomy", 'Care of Magical Creatures', 'Arithmancy', 'Flying']
			fig = plot_pairplot(datas._dataframe.drop(columns=col_to_remove), group=datas.classes_original)
			fig.write_html("viz/pair_plot.html")
	else:
		print("Error: arg must be:\n>>> python path/to/pair_plot.py dataset.csv")
