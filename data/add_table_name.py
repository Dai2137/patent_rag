# -*- coding: utf-8 -*-
import pandas as pd
import re
from pathlib import Path
from typing import Optional


def extract_table_name(path_string: str) -> Optional[str]:
    """
    path from 'result_X' extract and return X number

    Args:
        path_string: Example: '/mnt/eightthdd/raw_data/result_1/2/JP2010026792A'

    Returns:
        Number string (Example: '1') or None
    """
    match = re.search(r'result_(\d+)', path_string)
    if match:
        return match.group(1)
    return None


def extract_type_name(filename: str) -> Optional[str]:
    """
    Extract first character from CSV filename

    Args:
        filename: CSV filename (Example: 'A_data.csv')

    Returns:
        First character (Example: 'A') or None
    """
    return filename[0] if filename else None


def add_table_name_column(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Read CSV file and add table_name column and type column

    Args:
        csv_path: Input CSV file path
        output_path: Output CSV file path (None = overwrite)

    Returns:
        Processed DataFrame
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Add table_name column
    # df['table_name'] = df['path'].apply(extract_table_name)

    # Add type column from CSV filename's first character
    csv_filename = Path(csv_path).name
    df['type'] = extract_type_name(csv_filename)

    # Save
    if output_path is None:
        output_path = csv_path
    df.to_csv(output_path, index=False)

    return df


def process_all_csv_files(directory_path: str, pattern: str = "*.csv") -> None:
    """
    Process all CSV files in specified directory

    Args:
        directory_path: Directory path containing CSV files
        pattern: File search pattern (default: '*.csv')
    """
    dir_path = Path(directory_path)
    csv_files = list(dir_path.glob(pattern))

    print(f"Found {len(csv_files)} CSV files in {directory_path}")

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        df = add_table_name_column(str(csv_file))

        # Check: Display first 5 rows
        print(f"  - Rows: {len(df)}, Columns: {list(df.columns)}")
        print(f"  - Sample table_name values: {df['table_name'].unique()[:5]}")
        print()


if __name__ == "__main__":
    # Process all CSV files in path directory
    print("=== Process All CSV Files ===")
    process_all_csv_files("/home/sonozuka/staging/patent_rag/data/path")
