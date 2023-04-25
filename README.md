# 2022_Sgro_LIHC_tumour_supressors

A repository with computational code and outputs to accompany the manuscript Sgro *et al.* (Accepted 2023, *Clinical Epigenetics*), Epigenetic reactivation of tumor suppressor genes with CRISPRa technologies as precision therapy for hepatocellular carcinoma.

Analysis of the RNA-seq and DNAme data was performed by @jcursons and @MomenehForoutan. Please see the script folder for further details.

## Contact information

For information on the associated code please contact:
- Dr Joe Cursons (joseph.cursons (at) monash.edu)
- Dr Momeneh (Sepideh) Foroutan (momeneh.foroutan (at) monash.edu)


For further information on the manuscript or project please contact:
- Agustin Sgro (agustin.sgro (at) uwa.edu.au)
- A/Prof. Pilar Blancafort (pilar.blancafort (at) uwa.edu.au)

## Associated manuscript

The scientific manuscript associated with this repository has been accepted at *Clinical Epigenetics* and further information will be available soon.


## Data availability

Data generated for this project are available at GEO under accession code [GSE211837](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE211837).

### Public data

ENSEMBL reference for RNA-seq analysis:
- http://ftp.ensembl.org/pub/release-89/gtf/homo_sapiens/Homo_sapiens.GRCh38.89.gtf.gz  


## Project structure

### data

A folder containing intermediate output files used in this study and results from other reports. 

- mmc1.xlsx:
- new.c1.vs.c23.txt:
- new.c2.vs.c13.txt:
- new.c3.vs.c12.txt:


### script

- 2021_Sgro_HCC_EpiCRISPR.py:
- 850k_analysis.Rmd:


### figures

- Folder containing scripts and functions used for data analysis
- Unless otherwise stated please contact Assoc. Prof. Pilar Blancafort or Agustin Sgro for further information


## Dependencies

These scripts rely on some external dependencies.

### python - pysingscore

The python based implementation of singscore is used for gene set scoring in this manuscript. Installation instructions are available upon the Wiki: https://github.com/DavisLaboratory/PySingscore/wiki/Tutorial
