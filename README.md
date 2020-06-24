### VARITY Framework
VARITY is a data weighting approach to build specialized machine learning models. VARITY allows different weights to be placed on different training examples.  A high-quality core set of examples are given full weight, while examples from diverse add-on sets with potentially lower predictive utility are subjected to filtering and weighting. For each add-on set, examples are ordered by one or more quality-informative properties, and a threshold used to filter out the examples with low predictive utility, with a single weight assigned to all retained examples. Filtering thresholds and weights are treated as hyper-parameters and optimized for performance on the core set of examples using cross-validation.

To apply VARIYT framework:

1) Compose training data (both high-quality core set and add-on sets with uncertain quality).
2) Optional step: run moving window analysis on candidate quality-informative properties for each add-on set.
3) Config filtering and weighting hyper-parameters using quality-informative properties for each add-on set.
4) Run hyper-parameters tuning using cross-validation to determine weight for each add-on set.
5) Training the final VARITY model with core set and weighted add-on sets.

VARITY framework supports multiple "sessions" each has it own data and configuration.

First download https:/github/jowuca/varity/VARITY_exmaple.zip and extract to a local folder. The downloaded folder contains an existing session named "Final" for the manuscript "Improved pathogenicity prediction for rare human missense variants". Using the code in this git repository, you can explore the existing session or create a new session to build your own VARITY models. Please read VARITY Framework user guide before you start.  



   


