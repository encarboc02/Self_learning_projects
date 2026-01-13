# Analysis of Adverse Drug Reactions of Ozempic

## Description
This project performs a comprehensive analysis of adverse drug reactions (ADRs) reported for **Ozempic** using the FDA Adverse Event Reporting System (FAERS) dataset from July to September 2025.  
The goal is to identify patterns of serious and non-serious outcomes, explore patient demographics, concomitant medications, and provide an initial predictive analysis of factors contributing to severe adverse events.

--- 

## Objectives
- Explore demographic characteristics (age, sex) of patients reporting Ozempic use.
- Analyze the frequency and distribution of adverse reactions reported.
- Examine the relationship between the number of concomitant drugs and severe outcomes.
- Develop predictive models (Random Forest and Logistic Regression) to identify factors associated with serious adverse events.
- Visualize top adverse reactions for patients with and without serious outcomes.

--- 

## Workflow
1. **Data Acquisition**  
   Download FAERS datasets (DEMO, DRUG, REAC, OUTC) for the latest quarter.

2. **Data Preprocessing**  
   - Filter cases where Ozempic is the primary suspect drug (`role_cod='PS'`).  
   - Harmonize demographic variables: convert age to years and fill missing sex values as "Unknown".  
   - Aggregate counts of adverse events and concomitant medications per patient.

3. **Descriptive Analysis**  
   - Distribution of patient age and sex.  
   - Number of adverse events per patient.  
   - Number of concomitant medications per patient.  
   - Proportion of serious vs non-serious outcomes.  
   - Compare age, sex, and drug counts between serious and non-serious outcomes.

4. **Predictive Modeling**  
   - Random Forest Classifier to evaluate feature importance.  
   - Logistic Regression with class weighting to handle imbalanced outcomes.  
   - Evaluate performance using classification report, confusion matrix, and ROC-AUC.

5. **Adverse Reaction Analysis**  
   - Identify top adverse reactions for patients with serious outcomes.  
   - Compare proportions of reactions between serious and non-serious outcomes.

--- 

## Conclusions
  - The number of drugs combined with Ozempic is the most relevant factor in the dataframe for predicting serious outcomes.  
  - Adverse reactions (Emotional distress, Nausea, Dehydration, and Vomiting) could serve as risk indicators.  
  - Predictive models are subject to limitations due to the inherently noisy nature of FAERS data.
   
       

