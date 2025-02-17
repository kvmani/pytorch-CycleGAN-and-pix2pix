from bs4 import BeautifulSoup
import sys

def filter_epochs(input_html, output_html, start_epoch, step):
    # Read the input HTML file
    with open(input_html, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # Find all epoch headers and their corresponding tables
    epochs = soup.find_all('h3')
    tables = soup.find_all('table')

    # Prepare the new HTML content
    filtered_content = []
    for epoch, table in zip(epochs, tables):
        # Extract the epoch number from the header text (e.g., "epoch [10]")
        epoch_number = int(epoch.text.split('[')[1].split(']')[0])

        # Check if the epoch should be included
        if epoch_number >= start_epoch and (epoch_number - start_epoch) % step == 0:
            filtered_content.append((epoch, table))

    # Create a new HTML structure with filtered epochs
    new_html = BeautifulSoup('<!DOCTYPE html><html><head></head><body></body></html>', 'html.parser')
    new_html.head.append(soup.head.title)
    for epoch, table in filtered_content:
        new_html.body.append(epoch)
        new_html.body.append(table)

    # Write the new HTML to the output file
    with open(output_html, 'w', encoding='utf-8') as file:
        file.write(new_html.prettify())

if __name__ == "__main__":
    # Input arguments: input HTML file, output HTML file, start epoch, and step size


    input_html = "/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_9.0_batch_16_epoch_600_pool_100_vanilla/web/index.html"
    output_html ="/home/lus04/kvmani/ml_works/kaushal_2025/cyclegan/checkpoints/ebsd_data_9.0_batch_16_epoch_600_pool_100_vanilla/web/small_index.html"
    start_epoch = 0
    step = 10

    filter_epochs(input_html, output_html, start_epoch, step)
    print(f"Filtered HTML saved to {output_html}")
