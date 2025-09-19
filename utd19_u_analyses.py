import json

import numpy as np  # import numpy package under shorthand "np"
import pandas as pd  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
import sys
import os


# data = pd.read_csv('utd19_u/city_augsburg.csv')
# print(data.head(5))
# print(data.describe())
# print(data.isnull().sum())

def get_city_data(city):
	files = ["", "", ""]
	for file in os.listdir("utd19_u/"):
		if os.path.isfile(os.path.join("utd19_u/", file)) and city in file:
			if "detectors" in file:
				files[0] = file
			if "links" in file:
				files[1] = file
			if "utd19_u" in file:
				files[2] = file
	print(files)
	df_1 = pd.read_csv("utd19_u/" + files[0])
	df_2 = pd.read_csv("utd19_u/" + files[1])
	df_3 = pd.read_csv("utd19_u/" + files[2])
	df = pd.merge(df_1, df_2, on="linkid")
	df = pd.merge(df, df_3, on="detid")
	df = df.drop(columns=['citycode_y', 'citycode_x', 'long_x', 'lat_x', 'speed', 'fclass'])
	print(df.head(5))
	# print(df.describe())

	return [
		str(city),
		str(len(df.index)),
		str(len(df["day"].unique())),
		str(len(df["road"].unique())),
		str(len(df["detid"].unique())),
		str(df[['flow']].mean(axis=0)[0]),
		str(df[['flow']].std(axis=0)[0]),
		str(df[['flow']].median(axis=0)[0]),
		str(df[['occ']].mean(axis=0)[0]),
		str(df[['occ']].std(axis=0)[0]),
		str(df[['occ']].median(axis=0)[0]),
	]


def main():
	all_files = os.listdir("utd19_u/")
	names = []
	city_results = []
	for name in all_files:
		names.append(name.split("_")[0])
	unique_names = list(set(names))
	for unique_name in unique_names:
		force = True
		while force:
			try:
				if unique_name in ["london"]:
					break
				city_results.append(get_city_data(unique_name))
				force = False
			except IsADirectoryError:
				if unique_name in ["losanageles", "losangeles", "tokyo"]:
					break
				continue

	city_data = pd.DataFrame(city_results,
							columns=["city", "data_points", "days", "roads", "detectors", "flow_mean", "flow_std",
									"flow_median", "occ_mean", "occ_std", "occ_median"])
	return city_data


if __name__ == '__main__':
	data = main()
	print(data.head(50))
	data.to_csv("utd19_u_results/city_summary")


# city
# data_points
# days
# roads
# detectors
# flow_mean
# flow_std
# flow_median
# occ_mean
# occ_std
# occ_median
