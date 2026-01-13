# KRAS Sequence Analysis with Biopython

## Description
This project performs a basic bioinformatics analysis of the **KRAS** gene, one of the most studied proto-oncogenes in cancer, using Python and the **Biopython** library.

The workflow includes downloading sequences from public databases, analyzing basic metrics, extracting the coding region (CDS), translating to protein, and comparing across species through alignments and BLAST.

---

## Project Objectives
- Download the human KRAS protein sequence from **UniProt**.
- Calculate basic protein metrics (length, amino acid composition).
- Download the nucleotide sequence from **GenBank** and extract the CDS.
- Transcribe and translate the CDS to verify correspondence with UniProt.
- Calculate the GC percentage of the coding region.
- Compare the human protein with the mouse protein via global alignment.
- Perform a BLAST alignment against NCBI's **nr** database.
- Export significant alignments to a CSV file.

## 🧬 Workflow
1. Download the human KRAS protein (UniProt ID: `P01116`)
2. Analyze basic protein metrics
3. Download the coding sequence (GenBank ID: `NM_033360.4`)
4. Extract the CDS, transcribe, and translate
5. Verify equivalence between UniProt and GenBank
6. Human vs. mouse protein alignment (UniProt ID: `P32883`)
7. BLASTp against the `nr` database
8. Export results and calculate %GC
