import os
import csv  # Import the csv module

def split_alexa_list_csv(input_csv_filepath, output_dir):
    """Splits a CSV Alexa-like domain list (like top-1m.csv) into files based on the first letter."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist

    output_files = {}  # Dictionary to store file objects for each letter/digit

    try:
        with open(input_csv_filepath, 'r', encoding='utf-8', newline='') as csvfile: # Open as CSV, handle newlines
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if not row:  # Skip empty rows
                    continue
                try:
                    domain_rank, domain = row[0], row[1] # Assuming rank is in column 0, domain in column 1
                except IndexError: # Handle rows with fewer than 2 columns
                    print(f"Warning: Skipping row with insufficient columns: {row}")
                    continue

                domain = domain.strip() # Clean up domain, just in case
                if not domain:  # Skip empty domains
                    continue

                first_char = domain[0].lower()  # Get first char and lowercase it

                if 'a' <= first_char <= 'z':
                    output_filename = f"{first_char}.txt"
                elif '0' <= first_char <= '9':
                    output_filename = f"{first_char}.txt"
                else:
                    output_filename = "other.txt" # For domains starting with other chars

                output_filepath = os.path.join(output_dir, output_filename)

                if output_filename not in output_files:
                    output_files[output_filename] = open(output_filepath, 'w', encoding='utf-8') # Open file if not already open

                output_files[output_filename].write(domain + "\n")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_csv_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        for file_obj in output_files.values():
            file_obj.close()  # Close all opened files


if __name__ == "__main__":
    input_filepath = "./data/top-1m.csv"  # <-- Path to your top-1m.csv file
    output_directory = "./data/alexa-tld/"  # Output directory for letter-based files

    split_alexa_list_csv(input_filepath, output_directory)

    print(f"Alexa list (from CSV) split into files in: {output_directory}")