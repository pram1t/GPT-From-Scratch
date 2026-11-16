import yaml
from pathlib import Path

def load_config(config_path, selection=None):
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config[selection] if selection is not None else config



if __name__ == "__main__":
    #current_dir = Path.cwd()

    script_dir = Path(__file__).resolve().parents[2]
    print(" the dir is: ",script_dir)

    file_path = script_dir / "configs"/ "models" / "gpt_model.yaml"
    print(" the file is: ",file_path)

    config = load_config(file_path)
    model_config = load_config(file_path, selection="model")


    print(" the model config is: ",model_config)
    block_size = model_config["block_size"]
    n_layers = model_config["n_layer"]
    n_heads = model_config["n_head"]
    n_embd = model_config["n_embd"]
    dropout = model_config["dropout"]
    bias = model_config["bias"]

    print(block_size, n_layers, n_heads, n_embd, dropout, bias)