# Análisis de efectos adversos de Ozempic (FAERS Q3 2025)

## Descripción
Este proyecto analiza los reportes de efectos adversos relacionados con el fármaco **Ozempic** utilizando los datos públicos de **FAERS (FDA Adverse Event Reporting System, Q3 2025)**.

---

## Objetivos del proyecto
  - Explorar la distribución de efectos adversos y variables demográficas.
  - Identificar factores asociados a efectos graves ("serious").
  - Analizar las reacciones adversas más frecuentes en pacientes graves vs no graves.
  - Aplicar modelos predictivos (Random Forest y Regresión Logística) para evaluar la influencia de variables clínicas y demográficas.

---

## Archivos descargados utilizados
  - DEMO25Q3.txt (información demográfica)
  - DRUG25Q3.txt (Medicamentos reportados)
  - REAC23Q3.txt (Efectos adversos)
  - OUTC25Q3.txt (Resultados clínicos del paciente)

---

## Flujo de trabajo
  -  Importación y filtrado de datos
  -  Preparación de la base de datos
  -  Análisis descriptivo y visualización
  -  Modelado predictivo
  -  Análisis de reacciones adversas específicas

## Conclusiones (también en el script)
  - El número de fármacos combinados con Ozempic es el factor más relevante del Dataframe para predecir efectos graves
  - Las reacciones adversas (Emotional distress, Nausea, Dehydration y Vomiting) podrían servir como indicadores de riesgo
  - Los modelos predictivos están sujetos a limitación debido a la inherente naturaleza ruidosa de os datos FAERS
