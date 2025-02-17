import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_training_log(log_file):
    """
    Parses a CycleGAN-style training log file and returns a pandas DataFrame
    with columns:
      ['epoch', 'iters', 'time', 'data', 'D_A', 'G_A', 'cycle_A', 'idt_A',
       'D_B', 'G_B', 'cycle_B', 'idt_B'].
    """
    # Regex pattern to capture the relevant numeric fields.
    pattern = re.compile(
        r"\(epoch:\s*(\d+),\s*iters:\s*(\d+),\s*time:\s*([\d.]+),\s*data:\s*([\d.]+)\)\s*"
        r"D_A:\s*([\d.]+)\s*G_A:\s*([\d.]+)\s*cycle_A:\s*([\d.]+)\s*idt_A:\s*([\d.]+)\s*"
        r"D_B:\s*([\d.]+)\s*G_B:\s*([\d.]+)\s*cycle_B:\s*([\d.]+)\s*idt_B:\s*([\d.]+)"
    )

    records = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = pattern.search(line)
            if match:
                # Extract the capturing groups
                epoch_str, iters_str, time_str, data_str, \
                dA_str, gA_str, cycleA_str, idtA_str, \
                dB_str, gB_str, cycleB_str, idtB_str = match.groups()

                # Convert to appropriate types
                epoch = int(epoch_str)
                iters = int(iters_str)
                time_val = float(time_str)
                data_val = float(data_str)
                D_A = float(dA_str)
                G_A = float(gA_str)
                cycle_A = float(cycleA_str)
                idt_A = float(idtA_str)
                D_B = float(dB_str)
                G_B = float(gB_str)
                cycle_B = float(cycleB_str)
                idt_B = float(idtB_str)

                records.append({
                    "epoch": epoch,
                    "iters": iters,
                    "time": time_val,
                    "data": data_val,
                    "D_A": D_A,
                    "G_A": G_A,
                    "cycle_A": cycle_A,
                    "idt_A": idt_A,
                    "D_B": D_B,
                    "G_B": G_B,
                    "cycle_B": cycle_B,
                    "idt_B": idt_B
                })
            else:
                # You can print a warning if a line doesn't match the pattern
                # print(f"Warning: could not parse line: {line}")
                pass

    df = pd.DataFrame(records)
    return df

def plot_metrics(df):
    """
    Takes the parsed DataFrame and generates various line plots
    showing the model's training metrics over epochs/iterations.
    """
    # Sort by epoch then iters to ensure the data is sequential
    df = df.sort_values(by=["epoch", "iters"]).reset_index(drop=True)

    # For convenience, create a "global_step" that increments each row
    df["global_step"] = range(len(df))

    # Example 1: Plot Discriminator Losses (D_A, D_B) vs. global_step
    plt.figure(figsize=(8, 5))
    plt.plot(df["global_step"], df["D_A"], label="D_A")
    plt.plot(df["global_step"], df["D_B"], label="D_B")
    plt.xlabel("Global Step")
    plt.ylabel("Discriminator Loss")
    plt.title("Discriminator Losses Over Training")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Example 2: Plot Generator Losses (G_A, G_B) vs. global_step
    plt.figure(figsize=(8, 5))
    plt.plot(df["global_step"], df["G_A"], label="G_A")
    plt.plot(df["global_step"], df["G_B"], label="G_B")
    plt.xlabel("Global Step")
    plt.ylabel("Generator Loss")
    plt.title("Generator Losses Over Training")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Example 3: Plot Cycle Consistency Losses (cycle_A, cycle_B)
    plt.figure(figsize=(8, 5))
    plt.plot(df["global_step"], df["cycle_A"], label="cycle_A")
    plt.plot(df["global_step"], df["cycle_B"], label="cycle_B")
    plt.xlabel("Global Step")
    plt.ylabel("Cycle Consistency Loss")
    plt.title("Cycle Losses Over Training")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Example 4: Plot Identity Losses (idt_A, idt_B)
    plt.figure(figsize=(8, 5))
    plt.plot(df["global_step"], df["idt_A"], label="idt_A")
    plt.plot(df["global_step"], df["idt_B"], label="idt_B")
    plt.xlabel("Global Step")
    plt.ylabel("Identity Loss")
    plt.title("Identity Losses Over Training")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # You can add additional plots or further statistical analysis as needed.

def main():
    log_file = "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_5.0_epoch_1000_pool_150_vanilla/loss_log.txt"
    df = parse_training_log(log_file)
    if df.empty:
        print("No data was parsed. Check the log file and regex pattern.")
        return

    print("Parsed DataFrame head:\n", df.head(20))

    # Show basic statistics
    print("\nDescriptive statistics:\n", df.describe())

    # Plot the metrics
    plot_metrics(df)

if __name__ == "__main__":
    main()
