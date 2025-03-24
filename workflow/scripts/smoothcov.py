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
import glob
from pathlib import Path
import numpy as np
import gzip

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

def smooth_bins(bin_data, window_size):
    """Apply smoothing to the binned coverage data"""
    # Group bins by chromosome
    chrom_bins = defaultdict(list)
    for (chrom, bin_idx), data in bin_data.items():
        # Store bin_idx and avg_coverage
        if data['count'] > 0:
            avg_coverage = data['coverage_sum'] / data['count']
        else:
            avg_coverage = 0
        chrom_bins[chrom].append((bin_idx, avg_coverage, data['count']))
    
    # Sort bins within each chromosome
    for chrom in chrom_bins:
        chrom_bins[chrom].sort(key=lambda x: x[0])
    
    # Apply smoothing for each chromosome
    smoothed_data = {}
    for chrom, bins in chrom_bins.items():
        bin_indices = [b[0] for b in bins]
        coverages = [b[1] for b in bins]
        counts = [b[2] for b in bins]
        
        # Apply rolling window average
        smoothed_coverages = []
        for i in range(len(coverages)):
            # Get window start and end
            start = max(0, i - window_size//2)
            end = min(len(coverages), i + window_size//2 + 1)
            
            # Calculate weighted average based on bin counts
            if sum(counts[start:end]) > 0:
                weighted_sum = sum(coverages[j] * counts[j] for j in range(start, end))
                total_weight = sum(counts[start:end])
                smoothed_coverages.append(weighted_sum / total_weight)
            else:
                smoothed_coverages.append(0)
        
        # Store smoothed values
        for i, (bin_idx, _, count) in enumerate(bins):
            smoothed_data[(chrom, bin_idx)] = {
                'smoothed_coverage': smoothed_coverages[i],
                'count': count
            }
    
    return smoothed_data

def open_file(filename):
    """Open a file, detecting if it's gzipped or not"""
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')  # 'rt' mode for text reading from gzip
    else:
        return open(filename, 'r')

def main():
    # Snakemake objects
    input_file = snakemake.input.bed
    output_file = snakemake.output[0]

    # Get parameters from snakemake.params
    bin_size = snakemake.params.bin
    chunk_size = snakemake.params.chunk
    spike_threshold = snakemake.params.spike
    smooth_window = snakemake.params.smooth

    # Use defaultdict to collect bin data
    bin_data = defaultdict(lambda: {'coverage_sum': 0, 'count': 0})

    print(f"Processing {input_file} with bin size {bin_size}bp...")
    line_count = 0

    with open_file(input_file) as f:
        reader = csv.reader(f, delimiter='\t')

        for line in reader:
            line_count += 1

            # Extract columns from mosdepth output: chrom, start, end, coverage
            if len(line) >= 4:  # Ensure valid line
                chrom = line[0]
                start = int(line[1])
                end = int(line[2])
                coverage = float(line[3])
                
                # Apply spike threshold filter
                if coverage > spike_threshold:
                    coverage = spike_threshold
                
                # Calculate bin
                bin_idx = start // bin_size
                bin_key = (chrom, bin_idx)

                # Update bin data
                bin_entry = bin_data[bin_key]
                bin_entry['coverage_sum'] += coverage
                bin_entry['count'] += 1

            if line_count % chunk_size == 0:
                print(f"Processed {line_count:,} lines...")

    print(f"Processed total of {line_count:,} lines.")
    
    # Apply smoothing to the binned data
    print(f"Smoothing data with window size {smooth_window} bins...")
    smoothed_data = smooth_bins(bin_data, smooth_window)
    
    print("Writing output...")

    with open(output_file, 'w') as out:
        # Write header
        out.write("chrom,start,end,coverage,positions_count\n")

        # Sort bins by chromosome and position
        sorted_keys = sorted(smoothed_data.keys(), key=natural_sort_key)

        for bin_key in sorted_keys:
            chrom, bin_idx = bin_key
            bin_info = smoothed_data[bin_key]

            bin_start = bin_idx * bin_size
            bin_end = (bin_idx + 1) * bin_size

            # Write bin data with smoothed coverage
            out.write(f"{chrom},{bin_start+1},{bin_end},{bin_info['smoothed_coverage']:.2f},{bin_info['count']}\n")

    print(f"Done! Smoothed coverage data written to {output_file}")

if __name__ == "__main__":
    main()
