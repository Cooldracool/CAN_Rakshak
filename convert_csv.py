import csv


GAP = 0.001  # seconds


def read_csv(file_path):
    """
    Reads a CAN dataset CSV.
    Returns:
        rows: list of rows
        start_time: first timestamp
        end_time: last timestamp
    """
    rows = []

    with open(file_path, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue
            rows.append(row)

    start_time = float(rows[0][0])
    end_time = float(rows[-1][0])

    return rows, start_time, end_time


def shift_dataset(rows, offset):
    """
    Shifts only the timestamp column.
    """
    shifted = []

    for row in rows:
        new_row = row.copy()

        timestamp = float(new_row[0])
        timestamp += offset

        # preserve 6 decimal places
        new_row[0] = f"{timestamp:.6f}"

        shifted.append(new_row)

    return shifted


def main():

    dos_file = r"C:\Users\tarun\Downloads\CANShield-1B01\datasets\Demo_dataset\modified_dataset\DoS.csv"
    fuzzy_file = r"C:\Users\tarun\Downloads\CANShield-1B01\datasets\Demo_dataset\modified_dataset\fuzzy.csv"
    gear_file = r"C:\Users\tarun\Downloads\CANShield-1B01\datasets\Demo_dataset\modified_dataset\gear.csv"

    output_file = "Combined_dataset.csv"

    print("Reading datasets...")

    dos_rows, dos_start, dos_end = read_csv(dos_file)
    fuzzy_rows, fuzzy_start, fuzzy_end = read_csv(fuzzy_file)
    gear_rows, gear_start, gear_end = read_csv(gear_file)

    # ----------------------------------------------------------
    # Shift Fuzzy
    # ----------------------------------------------------------

    fuzzy_offset = (dos_end + GAP) - fuzzy_start

    fuzzy_rows = shift_dataset(fuzzy_rows, fuzzy_offset)

    fuzzy_end_shifted = fuzzy_end + fuzzy_offset

    # ----------------------------------------------------------
    # Shift Gear
    # ----------------------------------------------------------

    gear_offset = (fuzzy_end_shifted + GAP) - gear_start

    gear_rows = shift_dataset(gear_rows, gear_offset)

    # ----------------------------------------------------------
    # Merge
    # ----------------------------------------------------------

    combined = dos_rows + fuzzy_rows + gear_rows

    print("Writing output...")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(combined)

    print("Done!")
    print()
    print(f"DoS      : {len(dos_rows)} rows")
    print(f"Fuzzy    : {len(fuzzy_rows)} rows")
    print(f"Gear     : {len(gear_rows)} rows")
    print(f"Total    : {len(combined)} rows")
    print()
    print(f"Fuzzy offset = {fuzzy_offset:.6f} seconds")
    print(f"Gear offset  = {gear_offset:.6f} seconds")
    print()
    print(f"Output written to '{output_file}'")


if __name__ == "__main__":
    main()

    