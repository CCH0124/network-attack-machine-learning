import tensorflow as tf
def get_dataset(file_path, **kwargs):
  # https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
  return dataset

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

# `pack()` function will pack together all the columns
def pack(features, label):
# `tf.stack()` stacks a list of rank-R tensors into one rank-(R+1) tensor.
  return tf.stack(list(features.values()), axis=-1), label

def normalize_numeric_data(data, mean, std):
  return (data-mean)/std