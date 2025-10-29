PATH_TESTS_ROOT=.tests/integration
PATH_RESOURCES=$PATH_TESTS_ROOT/resources
PATH_RESULTS=$PATH_TESTS_ROOT/results

mkdir -p $PATH_RESOURCES/clair3_model

# Download the clair3 model
curl -L https://cdn.oxfordnanoportal.com/software/analysis/models/clair3/r1041_e82_400bps_sup_v420.tar.gz | tar -xz -C $PATH_RESOURCES/clair3_model --strip-components=1

# Donwload resources from Zenodo
wget -P $PATH_RESOURCES \
    https://zenodo.org/records/15688873/files/severus_vntrs.bed \
    https://zenodo.org/records/15688873/files/dna_r9.4.1_e8_sup@v3.3.tar.bz2 \
    https://zenodo.org/records/15688873/files/annotations.gtf.gz \
    https://zenodo.org/records/15688873/files/ref.fasta.gz \
    https://zenodo.org/records/15688873/files/annotations.gtf.gz.tbi \
    https://zenodo.org/records/15688873/files/highlights.bed \
    https://zenodo.org/records/15688873/files/reads.pod5 \
    https://zenodo.org/records/15688873/files/sample1_tumor.bam \
    https://zenodo.org/records/15688873/files/human_GRCh38_no_alt_analysis_set.trf.bed \
    https://zenodo.org/records/15688873/files/regions.bed \
    https://zenodo.org/records/15688873/files/hg38.chrom.sizes

# create the empty ENCODE blacklist file
touch $PATH_RESOURCES/encode_blacklist.bed

# create regulatory_regions.bed from highlights.bed, replacing hex colour with proper label
sed 's/#00FF00/test_region_1/' $PATH_RESOURCES/highlights.bed > $PATH_RESOURCES/regulatory_regions.bed

# If we skip basecalling, copy the sample basecalled file to the results directory
# and fake the QC rule output to prevent dorado usage
if [[ "${SKIP_BASECALLING}" != "true" ]]; then
    mkdir -p $PATH_RESOURCES/dorado
    curl -L https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.9.1-linux-x64.tar.gz | tar -xz -C $PATH_RESOURCES/dorado --strip-components=1
    # Decompress the model file into the same directory
    tar -xjf $PATH_RESOURCES/dna_r9.4.1_e8_sup@v3.3.tar.bz2 -C $PATH_RESOURCES
else
    PATH_BASECALL=$PATH_RESULTS/basecall_dorado/sample1
    PATH_QC=$PATH_RESULTS/pycoqc/sample1
    mkdir -p $PATH_BASECALL $PATH_QC
    mv $PATH_RESOURCES/sample1_tumor.bam $PATH_BASECALL
    touch $PATH_QC/sample1_tumor.html
fi
