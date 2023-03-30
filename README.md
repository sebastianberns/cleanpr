# Improved Precision and Recall on Clean Features

Compute the Precision and Recall measures between two manifolds built from different data sources (tensor, generator model, or data set). Raw image data is passed through an embedding model to compute ‘clean’ features. Check the [cleanfeatures documentation](https://github.com/sebastianberns/cleanfeatures) for a list of available embedding models (default: InceptionV3). Builds on code from [youngjung/improved-precision-and-recall-metric-pytorch](https://github.com/youngjung/improved-precision-and-recall-metric-pytorch).

## Setup

### Dependencies

- torch (Pytorch)
- numpy
- cleanfeatures ([sebastianberns/cleanfeatures](https://github.com/sebastianberns/cleanfeatures))

## Usage

Assuming that the repository is available in the working directory or Python path.

```python
from cleanpr import PR  # 1.

measure = PR('path/to/model/checkpoint/')  # 2.
measure.set_data_manifold(data_1)  # 3.
precision, recall = measure.precision_recall(data_2)  # 4.
```

1. Import the main class.
2. Create a new instance, providing a directory path of an embedding model. This can be either the place the model checkpoint is already saved, or the place it should be downloaded and saved to.
3. Calculate the reference manifold, providing the data samples (either as tensor, generator model, or data set).
4. Compute the measures, given a model data source (tensor of samples, generator model, or data set).

### PR class

```python
measure = PR(k=3, model_path='./models', model='InceptionV3', device=None, **kwargs)
```

- `k` (int): k-nearest neighbor parameter. Default: 3.
- `model_path` (str or Path object, optional): path to directory where model checkpoint is saved or should be saved to. Default: './models'.
- `model` (str, optional): choice of pre-trained feature extraction model. Options:
  - CLIP
  - DVAE (DALL-E)
  - InceptionV3 (default)
  - Resnet50
- `cf` (CleanFeatures, optional): an initialized instance of CleanFeatures. If set, all other arguments will be ignored.
- `device` (str or torch.device, optional): device which the loaded model will be allocated to. Default: 'cuda' if a GPU is available, otherwise 'cpu'.
- `kwargs` (dict): additional model-specific arguments passed on to `cleanfeatures`. See below.

#### CLIP model-specific arguments

- `clip_model` (str, optional): choice of pre-trained CLIP model. Options: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14 (default), ViT-L/14@336px

### Methods

The class provides three methods to process different types of input: a data tensor, a generator network, or a dataloader.

All methods return a tensor of embedded features [B, F], where F is the number of features.

#### precision_recall

Calculate Precision and Recall given a data source to be compared against the reference manifold. Returns a tuple of two floats.

```python
precision, recall = measure.precision_recall(input, **kwargs)
```

- `input` (Tensor or nn.Module or Dataset): data source to be compared against the reference manifold, can be different types (see above)
- `kwargs` (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method. See below.

#### get_manifold

Compute manifold given a data source. Returns a Manifold object with attributes 'features' and 'radii'.

```python
manifold = measure.get_manifold(input, **kwargs)
```

- `input` (Tensor or nn.Module or Dataset): data source to process, can be different types (see above)
- `kwargs` (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method. See below.

#### set_data_manifold

Build and save the data manifold for reference.

```python
measure.set_data_manifold(input, **kwargs)
```

- `input` (Tensor or nn.Module or Dataset): data source to process, can be different types (see above)
- `kwargs` (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method. See below.

#### compute_features

Compute features given a data source (can be of different types), handled by `cleanfeatures`. Return matrix of data features where rows are observations and columns are variables.

```python
features = measure.compute_features(input, **kwargs)
```

- `input` accepts different data types:
  - (Tensor): data matrix with observations in rows and variables in columns. Processed by `cleanfeatures.compute_features_from_samples()`
  - (nn.Module): pre-trained generator model with tensor output [B, C, W, H]. Processed by `cleanfeatures.compute_features_from_generator()`
  - (Dataset): data set with tensors in range [0, 1]. Processed by `cleanfeatures.compute_features_from_dataset()`
- `kwargs` (dict): additional data source-specific arguments passed on to the corresponding `cleanfeatures` method. See above.

#### compute_metric

Compute the individual metrics (Precision and Recall), given two manifolds. For precision, `manifold` is the data set and `subjects` the generated samples. For recall, `manifold` is the generated samples and `subjects` the data set.

```python
coverage = measure.compute_metric(manifold, subjects)
```

- `manifold` (Manifold): reference manifold to test against
- `subjects` (Manifold): manifold to evaluate

#### compute_coverage

Compute the manifold coverage, either for the Precision or Recall metric. For precision, 'radii' is the dataset radii. For recall, 'radii' is the generated samples radii.

```python
coverage = measure.compute_coverage(manifold.radii, distances)
```

- `radii` (numpy.ndarray): radii of samples in reference manifold
- `distances` (numpy.ndarray): pairwise distances between samples in reference manifold and subjects

#### compute_pairwise_distances

##### Data source-specific arguments

- Tensor of samples (`torch.Tensor`):
  - `batch_size` (int, optional): Batch size for sample processing. Default: 128
- Generator model (`torch.nn.Module`):
  - `z_dim` (int): Number of generator input dimensions
  - `num_samples` (int): Number of samples to generate and process
  - `batch_size` (int, optional): Batch size for sample processing. Default: 128
- Data set (`torch.utils.data.Dataset`):
  - `num_samples` (int): Number of samples to generate and process
  - `batch_size` (int, optional): Batch size for sample processing. Default: 128
  - `num_workers` (int, optional): Number of parallel threads. Best practice is to set to the number of CPU threads available. Default: 0
  - `shuffle` (bool, optional): Indicates whether samples will be randomly shuffled or not. Default: False


## References

Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019). Improved Precision and Recall Metric for Assessing Generative Models. *Advances in Neural Information Processing Systems*.

- [Paper (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html)
- [Repository (Official Tensorflow implementation)](https://github.com/kynkaat/improved-precision-and-recall-metric)
