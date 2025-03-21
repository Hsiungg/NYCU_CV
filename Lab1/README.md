# NYCU Computer Vision 2025 Spring Lab1

Student ID: 313551127

Name: 王翔 Hsiang, Wang

## Task Description

Lab1 uses ResNet as the backbone and performs image classification on a dataset of 100 types of plants and animals, with the metric being top-1 accuracy.

## Environment Setup

### 1. Clone the Repository

```sh
    git clone https://github.com/Hsiungg/NYCU_CV.git
    cd NYCU_CV/Lab1
```

### 2. Install Dependencies

### Recommended Python Version

It is recommended to use **Python 3.12.9** for this lab.

Use the provided **requirements.txt** to handle the environment package dependencies.

```sh
# Install required libraries and tools
pip install -r requirements.txt
```

### 3. Download the Dataset

You can download the dataset from the following link:
[Download link](<https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view?pli=1>)

After downloading, place the **hw1-data.tar.gz** file inside the cloned repository directory.

After placing the **dataset.tar.gz** file in the cloned directory, extract it using:

```sh
mkdir -p data
mv dataset.tar.gz data/
cd data
tar -xzvf hw1-data.tar.gz
```

### 4. Set Up Training and Testing Environment

To start training the model, use the following command:

```sh
python train.py --data_root ./data/ --output_dir ./output_dir --run_name ./log_run
```

#### Training Options

- --data_root: Path to the dataset directory (default: ./data).
- --output_dir: Path to the output directory for model weights.
- --run_name: Directory name for TensorBoard run name

To start testing the model and output **prediction.csv** file, use the following command:

```sh
python test.py --data_root ./data/ --output_dir ./output_dir --run_name ./log_run
```

#### Testing Options

- --data_root: Path to the dataset directory (default: ./data).
- --output_dir: Path to the output directory for **predict.csv** file
- --model_path: Path to trained model for testing. (/path/to/your/model/model.safetensors)
  
## Performance snapshots

A snap shot.
