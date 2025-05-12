import tensorflow_datasets as tfds

# This path should point to the base directory that contains the dataset folder
# For example, if your data is in ~/my_datasets/my_rlds_dataset/1.0.0/
# then 'data_dir' should be '~/my_datasets'
data_dir = "/path/to/my_datasets"

ds, info = tfds.load(
    name="my_rlds_dataset",  # Name of the dataset folder
    data_dir=data_dir,
    with_info=True,
)

for episode in ds["train"]:
    print(episode)
