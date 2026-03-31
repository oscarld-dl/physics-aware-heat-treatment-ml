# Thesis structure

## Main objective
Predict heat-treatment outcomes for AISI 9310 using data-driven models.

## Inputs
- carbon concentration
- HT1
- HT2
- quenching pressure
- grain size
- ferrite fraction
- pearlite fraction
- cryogenic temperature/time
- tempering temperature/time

## Outputs
- austenite fraction
- martensite fraction
- hardness
- distortion

## Workflow used in the thesis
1. FEM data generation
2. data preprocessing and node-to-point mapping
3. exploratory data analysis
4. regression baseline models
5. synthetic data expansion
6. feature importance / SHAP
7. FFNN models
8. cascade architecture
9. comparison with experiments

## Current limitations
- residual stress not studied
- simplified C-ring geometry
- limited dataset size
- mostly data-driven, limited physical constraints

## Upgrade direction
- causal awareness
- physics-aware constraints
- first focus on quenching
- carbon-dependent KM
- latent heat / energy consistency