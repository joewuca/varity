### VARITY Framework
VARITY is a data weighting approach to build specialized machine learning models. VARITY allows different weights to be placed on different training examples.  A high-quality core set of examples are given full weight, while examples from diverse add-on sets with potentially lower predictive utility are subjected to filtering and weighting. For each add-on set, examples are ordered by one or more quality-informative properties, and a threshold used to filter out the examples with low predictive utility, with a single weight assigned to all retained examples. Filtering thresholds and weights are treated as hyper-parameters and optimized for performance on the core set of examples using cross-validation.

## Steps to apply VARIYT framework:
1. Assemble training data (both high-quality core set and add-on sets with uncertain quality).
2. Optional step: Identify quality-informative properties for each add-on set using moving window analysis.
3. Config filtering and weighting hyperparameters for each add-on set.
4. Run hyperparameter optimization to determine flitering thresholds and weight for each add-on set.
5. Train the final VARITY model with core set and weighted add-on sets.

## To bulid VARITY models:
1.Download http://varity.varianteffect.org/downloads/VARITY_Final.zip and extract to a local folder. 
   The downloaded folder contains an existing session named "Final" for the manuscript "Improved pathogenicity prediction for rare human missense variants". Using   the code in this git repository, you can explore the existing session or create a new session to build your own VARITY models. Please read VARITY Framework user guide before you start.  

VARITY framework supports multiple "sessions" each has it own data and configuration.



   


