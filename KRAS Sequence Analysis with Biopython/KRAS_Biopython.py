# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 19:21:31 2025

@author: ecarb
"""

import requests
from Bio import SeqIO
from Bio import SeqUtils
from Bio import Entrez
from Bio import pairwise2
from Bio.SeqUtils import GC
from Bio.pairwise2 import format_alignment
from Bio.Blast import NCBIWWW, NCBIXML
import pandas as pd
import os

#Definimos el directorio del script
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

data_dir = os.path.join(script_dir, "data")
result_dir = os.path.join(script_dir, "results")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)


def descarga_fasta (uniprot_id, path):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "w") as file:
            file.write(response.text)
        print(f"El archivo fasta se ha descargado correctamente en {path}")
    else:
        raise ValueError(f"Error al descargar el archivo")
        
uniprot_id = "P01116"
path = os.path.join(data_dir, f"{uniprot_id}.fasta")

descarga_fasta(uniprot_id, path)

#Métricas básicas del archivo
with open (path, "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
       protein_id = record.id
       protein_seq = record.seq
       
print(protein_id)
print(protein_seq)

##Longitud
longitud = len(protein_seq)

##Composición aa
aa_com = {aa: protein_seq.count(aa) for aa in set(protein_seq)}

##Aminoácidos con propiedades químicas
hidrofóbico = "AVLIPFWM"
polar = "STNQCY"
cargado = "DEKR"

aa_hidro = sum(protein_seq.count(aa) for aa in hidrofóbico)
aa_polar = sum(protein_seq.count(aa) for aa in polar)
aa_carga = sum(protein_seq.count(aa) for aa in cargado)

#Descarga de la secuencia original de ADN
Entrez.email= 'enriquecb2002@gmail.com'

handle_2 = Entrez.efetch(db = "nucleotide", id="NM_033360.4", rettype = "gb", retmode= "text")
record= SeqIO.read(handle_2, "gb")
print(record)

gen_seq = record.seq
print(gen_seq)

##Extraemos solo la parte codificante del gen KRAS
for i in record.features:
    if i.type == 'CDS':
        cds_gen = i.extract(record.seq)

##Transcribimos y traducimos
gen_seq_molde = cds_gen.reverse_complement() #No se usa al ya venir por defecto el sentido correcto
print(gen_seq_molde)

gen_arn = cds_gen.transcribe()
print(gen_arn)

##Proteina
prot = gen_arn.translate()
print(prot)

##Comprpbamos que son iguales las proteínas. Tanto la de Uniprot, como la derivada de la secuencia de ADN de Genebank
print(len(prot))
print(len(protein_seq))

if str(prot).rstrip("*") == str(protein_seq): 
    print("Las secuencias de aminoácidos de ambas protínas son iguales")
else:
    print("Las secuencias no son iguales")
    
#Ambas pertenecen a la isoforma del gen KRAS: GTPase KRAS isoform a


#Alinemaiento de proteínas(LOCAL)
##Descargamos el fasta e la proteína KRAS de ratón
uni_ids = "P32883"
path2 = os.path.join(data_dir, f"{uni_ids}.fasta")
descarga_fasta(uni_ids, path2)

with open (path2, "r") as handle:
    for record in SeqIO.parse(handle, "fasta"):
       protein_id2 = record.id
       protein_seq2 = record.seq
       
## Hacemos el alinemaineto
alineamiento = pairwise2.align.globalxx(protein_seq,protein_seq2)
for a in alineamiento:
    print(format_alignment(*a))

#Alinemaiento con BLAST
human_fasta_path = os.path.join(data_dir, f"{uniprot_id}.fasta")

with open(human_fasta_path, "r") as f:
    human_fasta = f.read().strip()

resultados = NCBIWWW.qblast("blastp", "nr", human_fasta)

with open ("alin_res.xml", "w") as file:
    blast_results = resultados.read()
    file.write(blast_results)

#Leemos y analizamos 
res = open ("alin_res.xml", "r")
blast_record = NCBIXML.read(res)

E_VALUE_THRESH = 0.01
significant_align = []

for align in blast_record.alignments:
    for hsp in align.hsps:
        if hsp.expect < E_VALUE_THRESH:
            significant_align.append({
                "title": align.title,
                "length": align.length,
                "e_value": hsp.expect,
                "query": hsp.query,
                "match": hsp.match,
                "subject": hsp.sbjct
            })

df_align = pd.DataFrame(significant_align)

#Guardamos los resultados en un csv
output_path = os.path.join(result_dir, "align_results.csv")
df_align.to_csv(output_path, index=False)

#Contenido GC 
GC_cont = GC(cds_gen)
print(f"El porcentaje de GC de la secuencia codificante de KRAS es {GC_cont:.2f}%")
