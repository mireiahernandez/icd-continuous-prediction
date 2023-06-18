# temporal-modelling-icd

Temporal modelling ICD-9 code predictions.

This repository contains the code to perform temporal modelling of Electronic Health Record sequences (MIMIC-III dataset) for ICD-9 code prediction.

The code is based on the code for the HTDC (Ng et al, 2022) model. The contributions of this work is to adapt the model to make predictions at any point in time, inclduing: modified architecture and temporal Label-Wise Attention Network, novel evaluation scheme for multiple time-point prediction, and auxiliary tasks to improve temporal predictions.
