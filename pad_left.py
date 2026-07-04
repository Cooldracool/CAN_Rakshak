import csv

INPUT_FILE = r"C:\Users\tarun\Downloads\CANShield-1B01\datasets\Demo_dataset\modified_dataset\Combined_dataset.csv"
OUTPUT_FILE = r"C:\Users\tarun\Downloads\CANShield-1B01\datasets\Demo_dataset\modified_dataset\Combined_dataset0.csv"

with open(INPUT_FILE, "r", newline="") as infile, \
     open(OUTPUT_FILE, "w", newline="") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Copy header
    header = next(reader)
    writer.writerow(header)

    for row in reader:

        if not row:
            continue

        timestamp = row[0]
        can_id = row[1]
        dlc = row[2]

        # Last column is always the label
        flag = row[-1]

        # Everything between DLC and flag are the payload bytes
        data_bytes = row[3:-1]

        # Pad missing bytes with "00"
        while len(data_bytes) < 8:
            data_bytes.insert(0, "00")

        # If somehow more than 8 bytes exist, keep only first 8
        data_bytes = data_bytes[:8]

        new_row = [timestamp, can_id, dlc] + data_bytes + [flag]

        writer.writerow(new_row)

print(f"Done! Padded dataset saved as '{OUTPUT_FILE}'")