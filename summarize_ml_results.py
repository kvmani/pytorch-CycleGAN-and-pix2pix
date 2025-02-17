import os
import argparse
import logging
from bs4 import BeautifulSoup

def setup_logging(log_file=None, verbose=False):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to the log file. If None, logs are printed to console.
        verbose (bool): If True, set logging level to DEBUG, else INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(filename=log_file, level=log_level, format=log_format)
    else:
        logging.basicConfig(level=log_level, format=log_format)

def update_image_paths(soup, subfolder):
    """
    Update the image paths in the parsed BeautifulSoup object.
    
    Args:
        soup (BeautifulSoup): The parsed HTML content.
        subfolder (str): The subfolder where the images are located.
    
    Returns:
        BeautifulSoup: Modified BeautifulSoup object with updated image paths.
    """
    for img_tag in soup.find_all('img'):
        old_src = img_tag['src']
        # Prepend the subfolder path to the image source
        new_src = os.path.join(subfolder, 'web', old_src)
        img_tag['src'] = new_src
        logging.debug(f"Updated image path from {old_src} to {new_src}")
    return soup

def extract_latest_epoch(html_file, subfolder):
    """
    Extract the latest epoch's HTML content from an index.html file and update image paths.
    
    Args:
        html_file (str): Path to the index.html file.
        subfolder (str): Subfolder path to be prepended to image paths.
        
    Returns:
        str: HTML string containing the latest epoch's <h3> and <table> tags.
             Returns None if extraction fails.
    """
    try:
        with open(html_file, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Find the first <h3> tag which corresponds to the latest epoch
            latest_epoch = soup.find('h3')
            if latest_epoch:
                table = latest_epoch.find_next('table')
                if table:
                    # Update the image paths in the content
                    updated_soup = update_image_paths(soup, subfolder)
                    # Return the HTML string for the latest epoch
                    return str(latest_epoch) + str(table)
            logging.warning(f"No epoch information found in {html_file}.")
    except Exception as e:
        logging.error(f"Error reading {html_file}: {e}")
    return None

def create_summary_html(root_folder, output_file):
    """
    Create a summary.html file consolidating the latest epochs from all runs.
    
    Args:
        root_folder (str): Path to the 'checkpoints' root folder.
        output_file (str): Path where the summary.html will be saved.
        
    Returns:
        int: Number of .html files successfully processed.
    """
    summary_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Summary of Latest Epochs</title>
      <meta charset="UTF-8">
      <style>
        body { font-family: Arial, sans-serif; }
        h1 { text-align: center; }
        .run { margin-bottom: 50px; }
        table { width: 100%; table-layout: fixed; }
        td { padding: 10px; }
        img { width: 256px; }
      </style>
    </head>
    <body>
      <h1>Summary of Latest Epochs from All Runs</h1>
    """
    
    processed_count = 0  # Counter for successfully processed .html files
    
    # Iterate over each subfolder in the root_folder
    for subfolder in sorted(os.listdir(root_folder)):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            index_file = os.path.join(subfolder_path, 'web', 'index.html')
            if os.path.exists(index_file):
                logging.info(f"Processing file: {index_file}")
                latest_epoch_content = extract_latest_epoch(index_file, subfolder)
                if latest_epoch_content:
                    summary_content += f'<div class="run">\n<h2>Run: {subfolder}</h2>\n'
                    summary_content += latest_epoch_content + "\n</div>\n"
                    processed_count += 1
                else:
                    logging.warning(f"Failed to extract epoch from {index_file}.")
            else:
                logging.warning(f"index.html not found in {subfolder_path}.")
    
    summary_content += """
    </body>
    </html>
    """
    
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(summary_content)
        logging.info(f"Summary file created at: {output_file}")
    except Exception as e:
        logging.error(f"Failed to write summary file: {e}")
    
    return processed_count

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a summary HTML file from ML training runs.")
    parser.add_argument(
        '-r', '--root', 
        type=str, 
        required=True, 
        help="Path to the 'checkpoints' root folder."
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default=None, 
        help="Path to save the summary.html file. Defaults to 'summary.html' in root folder."
    )
    parser.add_argument(
        '-l', '--log', 
        type=str, 
        default=None, 
        help="Path to save the log file. If not provided, logs are printed to console."
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help="Enable verbose logging (DEBUG level)."
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set default output file if not provided
    output_file = args.output if args.output else os.path.join(args.root, 'summary.html')
    
    # Setup logging
    setup_logging(log_file=args.log, verbose=args.verbose)
    
    logging.info("Starting summary generation process.")
    logging.debug(f"Root folder: {args.root}")
    logging.debug(f"Output file: {output_file}")
    
    # Create summary.html and get the count of processed files
    processed_count = create_summary_html(args.root, output_file)
    
    logging.info(f"Process completed. Total .html files processed: {processed_count}")

if __name__ == "__main__":
    main()
