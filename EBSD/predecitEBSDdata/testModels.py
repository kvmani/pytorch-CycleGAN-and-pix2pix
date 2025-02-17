import subprocess

# Define variables
results_dir = "testing_inference_3"
dataroot = "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla/web/real_A_images/"
name = "ebsd_data_2.0_lr1e-4_pool_100_batch_32_vanilla"
model = "cycle_gan"
input_nc = 3
output_nc = 3
gpu_ids = -1
num_test = 20


# Create the arguments list using variables
args = [
    "--results_dir", results_dir,
    "--dataroot", dataroot,
    "--name", name,
    "--model", model,
    "--input_nc", str(input_nc),  # Convert int to str
    "--output_nc", str(output_nc),
    "--gpu_ids", str(gpu_ids),
    "--num_test", str(num_test),

]

# Run the script with arguments
subprocess.run(["python", "test.py"] + args)
