import numpy as np
import os
import tensorflow as tf
import datetime

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def create_dataset(file_pattern, batch_size, mode):
    """Creates tf.data Dataset Object to feed into model
    Args:
        file_pattern: str, file matterns to TFRecord data
        batch_size: int, batch size for training
        mode: tf.esitmator.ModeKeys (TRAIN/EVAL).
    
    Returns:
        tf.data Dataset object
    """
    def _parse(serialized_example):
        """Parse serialized example and return feature_dict and label
        Args:
            serialized_example: tf.example to parse
        Returns:
            Parsed features dictionary and label.
        """
        feature_map = {  # review these
            'fare': tf.io.FixedLenFeature([], tf.float32),
            'day': tf.io.FixedLenFeature([], tf.int64),
            'hour': tf.io.FixedLenFeature([], tf.int64),
            'pickup_latitude': tf.io.FixedLenFeature([], tf.float32),
            'pickup_longitude': tf.io.FixedLenFeature([], tf.float32),
            'dropoff_latitude': tf.io.FixedLenFeature([], tf.float32),
            'dropoff_longitude': tf.io.FixedLenFeature([], tf.float32),
            'passengers': tf.io.FixedLenFeature([], tf.float32)
        }
        
        parsed_example = tf.io.parse_single_example(
            serialized=serialized_example,
            features=feature_map
        )
        features = add_engineered(parsed_example)
        label = features.pop("fare")
        return features, label
    
    files = tf.io.gfile.glob(file_pattern)
    dataset = tf.data.TFRecordDataset(filenames=files, compression_type="GZIP")
    dataset = dataset.map(_parse)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat().shuffle(buffer_size=10*batch_size)
    
    dataset = dataset.batch(batch_size)
    return dataset


def add_engineered(features):
    """Add engineered features to dict
    Args:
        features: dict, dictionary of input features
    Returns:
        features: dict, dictionary with additional engineered features added
    """
    print(features)
    features["lat_diff"] = features["pickup_latitude"] = features["dropoff_latitude"]
    features["long_diff"] = features["pickup_longitude"] = features["dropoff_longitude"]
    features["euclidean"] = tf.math.sqrt(features["long_diff"]**2 + features["lat_diff"]**2)
    return features


def serving_input_fn():
    """Creates serving input receiver for EvalSpec.
    Returns:
        tf.estimator.export.ServingInputReceiver object containing placeholders and features.
    """
    inputs = {
        "day": tf.compat.v1.placeholder(
            dtype=tf.dtypes.int64, shape=[None], name="dayofweek"),
        "hour": tf.compat.v1.placeholder(
            dtype=tf.dtypes.int64, shape=[None], name="hourofday"),
        "pickup_longitude": tf.compat.v1.placeholder(
            dtype=tf.dtypes.float32, shape=[None], name="pickuplon"),
        "pickup_latitude": tf.compat.v1.placeholder(
            dtype=tf.dtypes.float32, shape=[None], name="pickuplat"),
        "dropoff_longitude": tf.compat.v1.placeholder(
            dtype=tf.dtypes.float32, shape=[None], name="dropofflon"),
        "dropoff_latitude": tf.compat.v1.placeholder(
            dtype=tf.dtypes.float32, shape=[None], name="dropofflat"),
        "passengers": tf.compat.v1.placeholder(
            dtype=tf.dtypes.float32, shape=[None], name="passengers")
    }
    features = add_engineered(inputs)
    return tf.estimator.export.ServingInputReceiver(features=features, receiver_tensors=inputs)


def train_and_evaluate(args):
    """Build tf.estimator.DNNRegressor and call train_and_evaluate loop.
    Args:
        args: dict, dictionary of command line arguments from task.py
    """
    feat_cols = [
        tf.feature_column.numeric_column('day'),
        tf.feature_column.numeric_column('hour'),
        tf.feature_column.numeric_column('pickup_latitude'),
        tf.feature_column.numeric_column('pickup_longitude'),
        tf.feature_column.numeric_column('dropoff_latitude'),
        tf.feature_column.numeric_column('dropoff_longitude'),
        tf.feature_column.numeric_column('passengers'),
        tf.feature_column.numeric_column('euclidean'),
        tf.feature_column.numeric_column('lat_diff'),
        tf.feature_column.numeric_column('long_diff')
    ]
    
    estimator = tf.estimator.DNNRegressor(
        feature_columns=feat_cols,
        hidden_units=args['hidden_units'].split(' '),
        model_dir=args['output_dir']
    )
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: create_dataset(file_pattern=args["train_data_path"],
                                        batch_size=args["train_batch_size"],
                                       mode=tf.estimator.ModeKeys.TRAIN),
        max_steps=300
    )
    
    exporter = tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=serving_input_fn
    )
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: create_dataset(file_pattern=args["eval_data_path"],
                                        batch_size=args["eval_batch_size"],
                                        mode=tf.estimator.ModeKeys.EVAL),
        exporters=exporter,
        steps=50
    )
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

