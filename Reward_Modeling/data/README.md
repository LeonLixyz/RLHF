Data Generator is used to augment eval dataset for the purpose of dyadic sampling.

1. It takes in a json dataset in the dataset directory, splits it into `train.json` and `eval.json`, 
2. It then runs augmentation on the eval for 512 datapoints, saving that into `augmented.json`, 
3. After which it does dyadic_batching and saves that into `joint_eval.json`

## To run the code

Go into data_generator.py, edit the dataset name of data to be evaluated and other settings, and run the code

```python -m data.data_generator```