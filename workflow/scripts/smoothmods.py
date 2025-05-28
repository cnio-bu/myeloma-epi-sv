#!/usr/bin/env python3

# /// script
# dependencies = [
#    "numpy>=1.0.0",
# ]
# ///

import argparse
import sys
import csv
from collections import defaultdict
import re

# Redirect output to Snakemake logs
sys.stderr=open(snakemake.log[0], "a+")
sys.stdout=open(snakemake.log[0], "a+")

# Improved natural sort key function
def natural_sort_key(key):
    chrom, bin_idx = key
    # Handle common chromosome naming patterns
    if chrom.startswith('chr'):
        chrom = chrom[3:]  # Remove 'chr' prefix

    # Extract digits for numeric sort
    match = re.match(r'(\d+|X|Y|MT|M|UN|LAMBDA|UNMAPPED)', chrom, re.IGNORECASE)
    if match:
        # Special chromosomes get high values
        if match.group(1).upper() == 'X':
            return (100, bin_idx)
        elif match.group(1).upper() == 'Y':
            return (101, bin_idx)
        elif match.group(1).upper() in ('M', 'MT'):
            return (102, bin_idx)
        elif re.match(r'UN|UNMAPPED', match.group(1), re.IGNORECASE):
            return (998, bin_idx)
        elif match.group(1).upper() == 'LAMBDA':
            return (999, bin_idx)
        else:
            # Numeric chromosomes
            try:
                return (int(match.group(1)), bin_idx)
            except ValueError:
                pass

    # Fallback: convert to string tuple for non-standard chromosome names
    return (str(chrom), bin_idx)

def main():
    # Snakemake objects
    input_file = snakemake.input.bed
    output_file = snakemake.output[0]

    # Get parameters from snakemake.params
    bin_size = snakemake.params.bin
    chunk_size = snakemake.params.chunk

    # Use defaultdict to avoid constant key checking
    bin_data = defaultdict(lambda: {'total_weight': 0, 'weighted_sum': 0, 'count': 0})

    print(f"Processing {input_file} with bin size {bin_size}bp...", file=sys.stderr)
    line_count = 0

    with open(input_file, 'r') as f:
        # Use csv reader instead of pandas for speed
        reader = csv.reader(f, delimiter='\t')

        for line in reader:
            line_count += 1

            # Extract only the columns we need (avoid creating full objects)
            chrom = line[0]
            start = int(line[1])
            coverage = int(line[9])  # Nvalid_cov
            percent_modified = float(line[10])

            # Calculate bin
            bin_idx = start // bin_size
            bin_key = (chrom, bin_idx)

            # Update bin data (faster than creating temp objects)
            bin_entry = bin_data[bin_key]
            bin_entry['total_weight'] += coverage
            bin_entry['weighted_sum'] += coverage * percent_modified
            bin_entry['count'] += 1

            if line_count % chunk_size == 0:
                print(f"Processed {line_count:,} lines...", file=sys.stderr)

    print(f"Processed total of {line_count:,} lines.", file=sys.stderr)
    print("Writing output...", file=sys.stderr)

    with open(output_file, 'w') as out:
        # Write header
        out.write("chrom,start,end,methylation_percent,cpg_count\n")

        # Use sorted with our improved natural_sort_key function
        sorted_keys = sorted(bin_data.keys(), key=natural_sort_key)

        for bin_key in sorted_keys:
            chrom, bin_idx = bin_key
            bin_info = bin_data[bin_key]

            # Calculate weighted average
            if bin_info['total_weight'] > 0:
                avg_methylation = bin_info['weighted_sum'] / bin_info['total_weight']
            else:
                avg_methylation = float('nan')

            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size

            # Write bin data
            out.write(f"{chrom},{bin_start+1},{bin_end},{avg_methylation:.2f},{bin_info['count']}\n")

    print(f"Done! Smoothed data written to {output_file}")

if __name__ == "__main__":
    main()