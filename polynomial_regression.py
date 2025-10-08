import sys
import os
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['agg.path.chunksize'] = 100


def make_smaller(city_name):
	files = ["", "", ""]
	for file in os.listdir("utd19_u/"):
		if os.path.isfile(os.path.join("utd19_u/", file)) and city_name in file:
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
	df = df[df["error"].isna()]
	df = df.drop(
		columns=['citycode_y', 'citycode_x', 'long_x', 'lat_x', 'speed', 'fclass', 'detid', 'linkid', 'road', 'order',
				 'piece', 'group', 'lanes', 'limit', 'city', "error"])
	# Uncomment to create a one week dataset
	start_date = "2015-01-01"
	end_date = "2015-01-07"
	date_mask = (df['day'] >= start_date) & (df['day'] <= end_date)
	df = df.loc[date_mask]

	# handled days cyclically within the week and within the year
	col = df.apply(lambda row: datetime.datetime.strptime(row.day, "%Y-%m-%d").weekday(), axis=1)
	df = df.assign(day_of_week=col.values)
	df["day_of_week_sin"], df["day_of_week_cos"] = make_data_cycle(df, "day_of_week", 7)

	col = df.apply(lambda row: datetime.datetime.strptime(row.day, "%Y-%m-%d").timetuple().tm_yday, axis=1)
	df = df.assign(day_of_year=col.values)
	df["day_of_year_sin"], df["day_of_year_cos"] = make_data_cycle(df, "day_of_year", 365)
	# handle intervals cyclically within the day
	df["interval_sin"], df["interval_cos"] = make_data_cycle(df, "interval", 60 * 60 * 24)

	df = df.drop(columns=["day", "day_of_week", "day_of_year", "interval"])
	return df


def make_data_cycle(df, row, period):
	sin_cycle = np.sin(2 * np.pi * df[row] / period)
	cos_cycle = np.cos(2 * np.pi * df[row] / period)
	return sin_cycle, cos_cycle


def get_features_and_labels(df, target_column="flow"):
	print(df.head(1))

	# X = df.drop(columns=["flow", "occ", "long_y", "lat_y", "pos", "length", "Unnamed: 0", "day_of_week_sin", "day_of_year_sin", "day_of_week_cos", "day_of_year_cos"])
	# X = df.drop(columns=["flow", "occ", "long_y", "lat_y", "pos", "length", "Unnamed: 0"])
	# X = df.drop(columns=["flow", "occ", 7"long_y", "lat_y", "Unnamed: 0"])
	X = df.drop(columns=["flow", "occ", "Unnamed: 0"])
	y = df[target_column]
	# show density of one feature
	# sns.kdeplot(df[target_column], fill=True)
	#
	# plt.xlabel(target_column)
	# plt.ylabel("Density")
	# plt.title(f"Density Plot of {target_column}")
	# plt.show()
	return X, y


def main(dataset, target_column="flow"):
	print(f"loading {dataset} dataset")
	df = pd.read_csv(f"utd19_u_results/{dataset}.csv")
	# Polynomial regression too heavy for my computer, but on a small dataset a degree of 3 showed a little
	# improvement on the R2 value
	features, labels = get_features_and_labels(df, target_column=target_column)
	for degree in range(1, 2):
		poly = PolynomialFeatures(degree=degree, include_bias=False)
		X_train, X_test, y_train, y_test = train_test_split(poly.fit_transform(features), labels, test_size=0.3, random_state=42)
		regr = LinearRegression()
		regr.fit(X_train, y_train)
		y_pred = regr.predict(X_test)
		tr_error = mean_squared_error(y_test, y_pred)
		print(f"\n degree of polinome: {degree}")
		print('The training error is: ', tr_error)
		print("R²:", r2_score(y_test, y_pred))

	# plt.figure(figsize=(8, 6))  # create a new figure with size 8*6
	# # create a scatter plot of datapoints
	# features, labels_occ = get_features_and_labels(df, target_column="occ")
	# plt.scatter(labels_occ, labels, color='b', s=8)
	#
	# # # plot the predictions obtained by the learnt linear hypothesis using color 'red' and label the curve as "h(x)"
	# # plt.plot(y_test, y_pred, color='r', label='h(x)')
	#
	# plt.xlabel('occ', size=15)  # define label for the horizontal axis
	# plt.ylabel('flow', size=15)  # define label for the vertical axis
	#
	# plt.title('Correlation between the two potential labels', size=15)  # define the title of the plot
	# plt.legend(loc='best', fontsize=14)  # define the location of the legend
	#
	# plt.show()  # display the plot on the screen


if __name__ == "__main__":
	algorithm = sys.argv[1]
	if algorithm == "make_smaller":
		curated_data = make_smaller(sys.argv[2])
		curated_data.to_csv(f"utd19_u_results/{sys.argv[3]}.csv")
	else:
		output = main(sys.argv[2], sys.argv[3])

# length
# pos
# long_y
# lat_y
# interval
# day_of_week
