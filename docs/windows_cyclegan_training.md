# CycleGAN Training on Windows with GPU

This guide explains each step required to train the CycleGAN model in this repository using a GPU on a Windows machine. It assumes no prior experience with deep learning tools. All commands can be executed in the Windows **Command Prompt** or **PowerShell** (unless stated otherwise). If you prefer using the Windows Subsystem for Linux (WSL) the commands remain the same.

## 1. Prepare the Dataset

CycleGAN expects **unpaired** images from two different domains. The repository uses a structure with `trainA`, `trainB`, `testA`, `testB`, etc. This guide extends it with a `val` split.

1. Create a folder for your dataset, for example `C:\datasets\my_cyclegan_data`.
2. Inside that folder create three subfolders: `trainA`,  `trainB`, `testA`, `testB`, `valA` amd `valB`.
3. The result should look like:

```
C:\datasets\my_cyclegan_data\trainA\  (images from domain A for training)
C:\datasets\my_cyclegan_data\trainB\  (images from domain B for training)
C:\datasets\my_cyclegan_data\testA\   (images from domain A for testing)
C:\datasets\my_cyclegan_data\testB\   (images from domain B for testing)
C:\datasets\my_cyclegan_data\valA\    (images from domain A for validation)
C:\datasets\my_cyclegan_data\valB\    (images from domain B for validation)
```

Place your JPG or PNG images in the respective folders. The code automatically scans these directories.

## 2. Install Python and Dependencies

1. Install **Python 3.8 or later** from [python.org](https://www.python.org/). During installation enable the option to add Python to your PATH.
2. Install **PyTorch** with GPU support. Visit [https://pytorch.org](https://pytorch.org) and follow the instructions for your CUDA version. For example, using `pip`:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install the remaining packages listed in `requirements.txt` (dominate, visdom, wandb) using `pip`:

```
pip install -r requirements.txt
```

(If you cloned the repository into `C:\projects\pytorch-CycleGAN-and-pix2pix`, run the above command in that directory.)

4. Optionally verify that PyTorch detects your GPU by running:

```
python test_GPU_env.py
```

The script prints the detected GPU devices and CUDA version.

## 3. Configure Training Parameters

Training options are defined in [`options/base_options.py`](../options/base_options.py) and [`options/train_options.py`](../options/train_options.py). For typical CycleGAN training you mainly need to set:

- `--dataroot` – path to the dataset directory.
- `--name` – a folder name where checkpoints and HTML results will be stored under `checkpoints/`.
- `--model` – set to `cycle_gan` for unpaired training.
- `--gpu_ids` – which GPU id(s) to use, e.g. `0`.

Other useful parameters include batch size (`--batch_size`), number of epochs (`--n_epochs` and `--n_epochs_decay`), and image preprocessing settings (`--load_size`, `--crop_size`). Defaults are defined in the options files and can be overridden on the command line.

## 4. Start Training

1. Open **Command Prompt** (or PowerShell) and navigate to the repository folder. Example:

```
cd C:\projects\pytorch-CycleGAN-and-pix2pix
```

2. Launch training with a command like the following. Adjust the `dataroot` path and experiment `name` to your setup. The example uses GPU id 0.

```
python train.py --dataroot C:\datasets\my_cyclegan_data --name my_cyclegan_run --model cycle_gan --gpu_ids 0
```

During training the script prints losses every few iterations and saves generated images and models under `checkpoints\my_cyclegan_run`. If you want to monitor progress visually you can run the Visdom server in another terminal before starting training:

```
python -m visdom.server
```

Then open `http://localhost:8097` in a browser to view the training curves and example translations.

## 5. Understanding the Code Structure

The repository is organized so that `train.py` orchestrates the process. It uses helper modules in the `options`, `models`, and `data` packages:

- **`options`** – parses command line arguments and sets up the experiment (GPU ids, dataset, model type, etc.). Functions in [`BaseOptions`](../options/base_options.py) and [`TrainOptions`](../options/train_options.py) handle this.
- **`data`** – provides dataset classes. For unpaired images [`unaligned_dataset.py`](../data/unaligned_dataset.py) loads domain A and B images and applies preprocessing like resizing and cropping.
- **`models`** – contains implementations of different architectures. [`cycle_gan_model.py`](../models/cycle_gan_model.py) defines the generators and discriminators, cycle-consistency loss, and optimization steps.
- **`util`** – utility functions for logging, visualization, and HTML generation.

The high-level workflow in [`train.py`](../train.py) is:

1. Parse command line options and create the dataset.
2. Build the CycleGAN model with generators and discriminators.
3. For each epoch:
   - Iterate over the dataset, running forward and backward passes.
   - Display and log losses using Visdom and HTML files.
   - Save model checkpoints periodically.
4. After all epochs finish, final models are stored in `checkpoints/<name>`.

The design allows other models (e.g., pix2pix) to reuse the same training loop by providing different model classes and dataset handlers.

## 6. Testing the Trained Model

After training completes you can test the model on the validation or test folders:

```
python test.py --dataroot C:\datasets\my_cyclegan_data --name my_cyclegan_run --model cycle_gan --gpu_ids 0
```

Results are written to `results\my_cyclegan_run\latest_test`. Open the generated `index.html` file in a browser to view translations.

## 7. Additional Tips

- To continue training from a saved checkpoint, add `--continue_train`.
- To change image sizes or cropping behavior, adjust `--load_size`, `--crop_size`, and `--preprocess` options.
- Set `--no_dropout` if you do not want dropout layers in the generator.
- If you encounter out-of-memory errors, reduce `--batch_size` or image resolution.

This step-by-step approach should enable a beginner to prepare the dataset, configure parameters, and run unpaired CycleGAN training on a Windows machine with GPU support.

