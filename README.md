# Using Tensorflow with GCP AI-platform

This is quite an involved process, with quite a few high-level steps.
These steps were hard learned, so I wanted to solidify that process by documenting.

1. Training
    1. use a beam batch pipeline. This can be done locally (DirectRunner) or on cloud (Dataflow)
        - in conjunction with tensorflow and tensorflow_transform library
        - extract data from source (probably BigQuery)
            1. for training
            1. and evaluation
        - apply feature engineering transformations. Training will learn data parameters, and use this when transforming eval dataset.
        It's important to use tensorflow transformations rather than just regular beam transformations.
        Remember TensorFlow creates a DAG similar to Dataflow, but these are unrelated, so keep them unrelated mentally. 
        - write the datasets in TFRecord format to local/GCS folder
        - write the pre-processing transform function to local/GCS folder as well
    1. run a tensorflow training job
        - define a task file that lists args for parameters
        - use this to pass appropriate args to the model file. This can be run locally on small datasets, or as a job on cloud
        - this ultimately exports weights and a serving function to local file/GCS storage
2. Serving
    - can be done locally:
        1. ```bash
            echo -n '{"feat1": 50, "feat2": "str"}' >> /tmp/test.json
            gcloud ai-platform local predict \
                  --model-dir=${PWD}/my_model/export/exporter/157989465845 \
                  --json-instances=/tmp/test.json
            ```
    - can be done in the cloud:
        1. Create a model in GCP AI platform
        1. Create a version of the model, by pointing it at the GCS location of the model
        1. Use gcloud command line tool run make inferences (predictions) from data:
            ```bash
            gcloud ai-platform predict --model=mymodel --version=v1 --json-instances=/tmp/test.json 
           ```
           
The files in this directory do the following things.

[notebook.ipynb](./notebook.ipynb)

Looks more complex than it is.
First 50 lines are just setup, installing libs and setting variables.

Then a function to create slightly different queries for train/eval/test data.

Then a beam pipeline, with a few notable differences to the norm.

We import the following:
```python
from tensorflow_transform.tf_metadata import dataset_metadata, dataset_schema
```

Then we define the raw_data_schema, a dictionary with keys for columns, and values like so:

```python
dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation())
```

We should use `tf.int64` or `tf.float32` instead if appropriate.

We then create a special metadata object from this like so:

```python
raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema(raw_data_schema))
```

This will be used later to combine with PCollections, (presumably to provide more info to tensorflow created transforms, and help its DAG.)

With a pipeline defined, we need an additional import:
```python
from tensorflow_transform.beam import impl as beam_impl
```

Our initial setup should look like so:

```python
with beam.Pipeline(runner, options=opts) as p:
    with beam_impl.Context(temp_dir=dir_tmp):
        # do pipeline stuff...
```

There are a few important beam steps provided by tensorflow:
- `raw_data_metadata | 'Write Input Metadata' >> tft_beam_io.WriteMetadata(dir_rawdata_md, pipeline=p)`
- `raw_data = p | "ReadBQ" >> beam.io.Read(beam.io.BigQuerySource(query=create_query(1, EVERY_N), use_standard_sql=True))`
    This needs to be combined with the metadata, to form a dataset object that can be used with other tensorflow provided transforms, like so:
    ```python
    raw_dataset = (raw_data, raw_data_metadata)
    ```
- Same idea for reading eval dataset, combine with metadata
- `transformed_dataset, transform_fn = (raw_dataset | "Analyse and Transform Train" >> beam_impl.AnalyzeAndTransformDataset(preprocess_tft))`
    This is possibly the most important part. This is where input rows are accepted, scaling of feature columns happens, and feature engineering happens.
    Importantly, this returns the transformed dataset, but also the transform function, which knows how to appropriately scale the data.
    This is needed in order to transform the evaluation dataset.
    Note: the dataset returned comes in 2 parts, data, and metadata, and will need to be unpacked:
    ```python
    transformed_data, transformed_metadata = transformed_dataset
    ```
- `transformed_test_dataset = (raw_test_dataset, transform_fn) | "Transform Test" >> beam_impl.TransformDataset()` - notice this no longer needs the preprocess_tft input. It uses the optimised function learned from analysing the train dataset.
- `transform_fn | "Write Transform Function" >> transform_fn_io.WriteTransformFn(dir_md)` - this is crucial for serving time transformations


[task.py](./task.py)

Accepts arguments needed to start a training job.
Has no concept of whether it's for local running or the cloud.

[my_model.py](./my_model.py)

This is where the magic of ML happens.
Looks complex, but it can be boiled down quite rapidly to the last function.

`train_and_evaluate` is the whole objective. Define :
- the feature columns
- model type and hyper parameters
- training details, including where to get the input from, and how many steps
- serving function for taking raw input, adding feature engineered columns
- evaluation details, like where to get the dataset, serving_function to use
- finally, call train_and_evaluate on the model.

If you understand that function, the rest is small potatoes.
