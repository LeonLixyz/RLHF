import random
import json

def split_and_save_data(config):

    original_data_dir = config['data_dir']
    data_dir_formatted = config['data_dir'].format(config['dataset_name'])

    # Load the data
    with open(data_dir_formatted, 'r') as f:
        annotated = json.load(f)

    # Shuffle and split the data
    random.shuffle(annotated)
    dataset_len = len(annotated)
    eval_len = config["eval_len"]
    train_len = dataset_len - eval_len
    train_annotated = annotated[:train_len]
    eval_annotated = annotated[train_len:]

    # Save the split data to separate JSON files
    with open(data_dir_formatted.replace(".json", "_train.json"), 'w') as f:
        json.dump(train_annotated, f, ensure_ascii=False, indent=4)

    with open(data_dir_formatted.replace(".json", "_eval.json"), 'w') as f:
        json.dump(eval_annotated, f, ensure_ascii=False, indent=4)

    config['data_dir'] = original_data_dir

    
def main():
    config_path = "configs/vanilla_reward.json"
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)

    # set seed:
    random.seed(42)
    split_and_save_data(config)


if __name__ == "__main__":
    main()