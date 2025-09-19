# machine_learning_2025_aalto
Project for ML course

# Set up python environment:

We are using `pipenv` for dependency management, doc for setting it up there:
```
https://pipenv.pypa.io/en/latest/
```

# Split dataset

If a dataset is too big(long loading time and all), there is python script to create smaller datasets but splitting it based on the value of one of its headers

```
python3 -m split_dataset dataset_location column_name
for example:
python3 -m split_dataset dataset/utd19_u.csv city
```
