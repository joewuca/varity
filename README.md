## VARITY Framework
VARITY is a supervised machine learning approach to build specialized predictive models using training examples with optimized differential weights. Training examples are assembled into different training sets: 1) one core set of examples which are known to have high quality. 2) one or more add-on sets of examples with uncertain quality. For each training set, the weights of examples are determined using one or more logistic functions each takes one quality-informative property as input. The parameters of each logistic function are treated as hyper-parameters and optimized for performance on the core set of examples using cross-validation.

### Steps to apply VARITY framework:
1. Assemble training examples (high-quality core set and add-on sets with uncertain quality).
2. Identify quality-informative properties for each add-on set using domain knowledge (Optional: verify using moving window analysis).
3. Config hyper-parameters based on quality-informative properties (parameters of corresponding logistic functions).  
4. Optional: Run nested cross-validation to evaluate performance.
5. Run hyper-parameter optimization to determine weight of each training example. 
6. Train the final VARITY model with core set and weighted add-on sets.

To apply VARITY framework, please read [VARITY user guide](https://github.com/joewuca/varity/tree/master/VARITY_user_guide.pdf) and use  [VARITY python scripts](https://github.com/joewuca/varity/tree/master/python). 

### Technical Support
Please contact joe.wu.ca@gmail.com for technical support