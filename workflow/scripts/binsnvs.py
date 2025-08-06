import pandas as pd
import os
import gzip
import sys
from natsort import natsorted

# Redirect output to Snakemake logs
sys.stderr = open(snakemake.log[0], "a+")
sys.stdout = open(snakemake.log[0], "a+")

def open_vcf_file(vcf_file):
    """Open VCF file, handling both compressed (.gz) and uncompressed files"""
    if vcf_file.endswith('.gz'):
        return gzip.open(vcf_file, 'rt', encoding='utf-8')
    else:
        return open(vcf_file, 'r')

def process_vcf(vcf_file):
    """Extract PASS variants from VCF file"""
    filtered_rows = []
    
    with open_vcf_file(vcf_file) as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            columns = line.strip().split('\t')
            if len(columns) >= 7 and columns[6] == 'PASS':
                # Add chromosome, position and filter to list
                filtered_rows.append((columns[0], int(columns[1]), columns[6]))
    
    return pd.DataFrame(filtered_rows, columns=['chromosome', 'position', 'filter'])

def bin_variants(df, chrom_sizes_df, bin_size):
    """Bin variants based on genomic position"""
    intervals = []
    
    for chrom, group in df.groupby('chromosome'):
        # Get chromosome end position
        chrom_match = chrom_sizes_df.loc[chrom_sizes_df['chr'] == chrom]
        if len(chrom_match) == 0:
            print(f"Warning: Chromosome {chrom} not found in chromosome sizes file")
            continue
            
        chrom_end = chrom_match['end'].values[0]
        
        # Create bins
        start = 1
        end = bin_size
        
        while start <= chrom_end:
            count = group[(group['position'] >= start) & (group['position'] <= end)].shape[0]
            intervals.append((chrom, start, min(end, chrom_end), count))
            start += bin_size
            end += bin_size
    
    return pd.DataFrame(intervals, columns=['chr', 'start', 'end', 'value'])

def main():
    # Snakemake objects
    input_file = snakemake.input.vcf
    output_file = snakemake.output[0]

    # Get parameters from snakemake.params
    bin_size = snakemake.params.bin
    chrom_sizes_file = snakemake.params.chrom_sizes
    
    print(f"Processing: {input_file}")
    
    # Check file format
    if input_file.endswith('.gz'):
        print("Detected compressed VCF file (.gz)")
    else:
        print("Detected uncompressed VCF file")
    
    # Load chromosome sizes - THIS IS THE MODIFIED PART
    chrom_sizes_df = pd.read_csv(chrom_sizes_file, sep='\t', header=None, names=['chr', 'end'])
    # Add start column (always 1)
    chrom_sizes_df['start'] = 1
    
    # Process VCF file
    variants_df = process_vcf(input_file)
    
    # Bin variants
    intervals_df = bin_variants(variants_df, chrom_sizes_df, bin_size)
    
    # Sort chromosomes naturally
    intervals_df['chr'] = pd.Categorical(intervals_df['chr'], 
                                      categories=natsorted(intervals_df['chr'].unique()), 
                                      ordered=True)
    intervals_df = intervals_df.sort_values(['chr', 'start'])
    
    # Save to CSV
    intervals_df.to_csv(output_file, index=False)
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    main()