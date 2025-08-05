# Oxford Nanopore Methylation and Structural Variant Analysis Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15168660.svg)](https://doi.org/10.5281/zenodo.15168660)
[![Integration test](https://github.com/cnio-bu/myeloma-epi-sv/actions/workflows/integration-test.yaml/badge.svg)](https://github.com/cnio-bu/myeloma-epi-sv/actions/workflows/integration-test.yaml)

## Overview

This is a Snakemake workflow for analyzing Oxford Nanopore Technologies (ONT) sequencing data, with a focus on methylation calling and structural variant detection. This pipeline processes raw ONT reads (POD5 format) through basecalling, alignment, methylation analysis, and various structural variant calling methods.

## Features

- **Basecalling and QC**: Uses Dorado for basecalling with modified base detection, followed by quality assessment with PycoQC
- **Alignment**: High-quality read mapping with Minimap2
- **Methylation Analysis**:
  - Methylation pileup with ONT Modkit
  - Visualization with methylartist
- **Structural Variant Detection**: Multiple SV callers for comprehensive detection
  - Clair3 for SNV/indel calling
  - ClairS for somatic variant detection in tumor-normal pairs
  - Sniffles2 for germline structural variation detection
  - Severus for germline and somatic structural variation detection
- **Copy Number Aberration Analysis**: Uses Wakhan for copy number estimation

## Requirements

- Snakemake (>=7.0)
- Conda or Mamba (for environment management)
- For GPU-accelerated basecalling: CUDA-compatible GPU

## Installation

1. Clone this repository:
```bash
git clone https://github.com/cnio-bu/myeloma-epi-sv.git
cd myeloma-epi-sv
```
2. Create your config file:
```bash
cp config/config.yaml.example config/config.yaml
```
3. Edit the configuration file to match your environment, data locations, and analysis parameters.

## Configuration

The pipeline is configured through the `config/config.yaml` file, which includes:

- **Reference files**: Paths to genome reference, annotations, and other required files
- **Tool paths**: Locations of external tools like dorado and Clair3 models
- **Analysis parameters**: Quality thresholds and analysis options
- **Samples structure**: Hierarchical organization of samples with associated metadata
- **Resource specifications**: Resource allocation for different workflow steps

See `config/config.yaml.example` for a detailed example of the configuration structure.

## Usage

### Basic Execution

Run the full pipeline with:
```bash
snakemake --use-conda --cores <N>
```

### Execution with Slurm

For cluster environments using Slurm:
```bash
snakemake --use-conda --profile slurm
```

## Pipeline Steps

1. **Basecalling**: Convert raw POD5 files to BAM format with Dorado, including modified base detection
2. **Quality Control**: Generate QC metrics and reports with PycoQC
3. **Quality Filtering**: Filter reads based on quality score
4. **Alignment**: Map reads to reference genome with Minimap2
5. **Coverage Analysis**: Generate coverage statistics with Mosdepth and samtools
6. **Methylation Analysis**:
   - Extract methylation information with Modkit
   - Visualize methylation patterns with methylartist
7. **Variant Calling**:
   - SNV/indel detection with Clair3
   - Somatic variant detection with ClairS (for tumor-normal pairs)
   - Germline structural variant detection with Sniffles2
   - Germline/somatic structural variation detection with Severus
8. **Copy Number Aberration Analysis**: Generate haplotype-specific coverage plots with Wakhan

## Output Structure

The pipeline generates results in a hierarchical directory structure:

```
results/
├── basecall_dorado/       # Basecalled reads
├── pycoqc/                # Quality control reports
├── minimap2/              # Aligned reads
├── primary/               # Filtered primary alignments
├── mosdepth/              # Coverage statistics
├── modkit/                # Methylation data
├── methylartist/          # Methylation visualizations
├── clair3/                # Small variant calls
├── sniffles/              # Structural variant calls
├── severus/               # Tandem repeat expansions
├── clairs/                # Somatic variants
└── wakhan/                # Copy number aberrations
```

## Testing

Run the included integration test to verify that the pipeline is properly installed:

```bash
export SKIP_BASECALLING=true #skip this line to run the GPU-based basecalling step
bash .tests/integration/get_resources.sh
snakemake --sdm conda --cores 1 --directory .tests/integration
```
