# KRAS Sequence Analysis with Biopython

## Descripción
Este proyecto realiza un análisis bioinformático básico del gen **KRAS**, uno de los proto-oncogenes más estudiados en cáncer, utilizando Python y la librería **Biopython**.

El flujo de trabajo incluye la descarga de secuencias desde bases de datos públicas, el análisis de métricas básicas, la extracción de la región codificante (CDS), la traducción a proteína y la comparación entre especies mediante alineamientos y BLAST.

---

## Objetivos del proyecto
- Descargar la secuencia proteica humana de KRAS desde **UniProt**.
- Calcular métricas básicas de la proteína (longitud, composición de aminoácidos).
- Descargar la secuencia nucleotídica desde **GenBank** y extraer la CDS.
- Transcribir y traducir la CDS para comprobar su correspondencia con UniProt.
- Calcular el porcentaje de GC de la región codificante.
- Comparar la proteína humana con la proteína de ratón mediante alineamiento global.
- Realizar un alineamiento BLAST contra la base de datos **nr** del NCBI.
- Exportar los alineamientos significativos a un archivo CSV.


## 🧬 Flujo de trabajo
1. Descarga de la proteína KRAS humana (UniProt ID: `P01116`)
2. Análisis de métricas proteicas básicas
3. Descarga de la secuencia codificante (GenBank ID: `NM_033360.4`)
4. Extracción de la CDS, transcripción y traducción
5. Verificación de equivalencia entre UniProt y GenBank
6. Alineamiento proteína humana vs ratón (UniProt ID: `P32883`)
7. BLASTp contra la base de datos `nr`
8. Exportación de resultados y cálculo del %GC
