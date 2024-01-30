## LISAR: Influential assets in Large-Scale Vector AutoRegressive Models
This repository contains the code which implements the algorithm for the LISAR model introduced in the paper [*"Influential assets in Large-Scale Vector AutoRegressive Models"*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619531) as well as the high-frequency data for the financial companies contained in the S&P100 used in the analysis of said paper. 

### Code

The code consists of 9 files:

1. `main_LISAR.R`

2. `LISAR_LASSO.R`

3. `LISAR_SCAD.R`

4. `LISAR_AdapLASSO.R`

5. `LISAR_LASSO2.R`

6. `LISAR_lambda_select.R`

7. `LISAR_alpha_select.R`

8. `LISAR_helper_functions.R`

9. `LISAR_evaluation.R`

The first code file contains necessary procedures to run the algorithm as described in the paper as well as loads the data and sets the control parameters. The second, third and fourth code files are internal code files which contain the algorithms for the LISAR.LASSO, LISAR.SCAD, LISAR.AdapLASSO models. The fifth code file contains an implementation of a VAR with LASSO penalty. The sixth and seventh code files contain the procedures to select the optimal lambdas and alphas. For details of these parameters, please see the paper. The eights code file contains helper functions. The ninth one the evaluation code for the optimal model. For the details of the LISAR model and algorithm, please see the LISAR paper: [view paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619531)


### Data 

The repository contains a .RData files which contains the 5 minute price observations of the finance stocks contained in the S&P100. 


### Output

The code returns a list with 3 entries which contain the following information: 

- `EvaluateModel$Model' : Optimal LISAR model
- `EvaluateModel$Eval.Model' : Evaluation parameters of the optimal LISAR model
- `EvaluateModel$Lambdas.Model' : Lambdas, alpha and gamma for optimal LISAR model


### Reference

When using the code or data in your own work, please reference to our paper. Thank you in advance!: 

Zhang, K., Trimborn, S., (2023) Influential assets in Large-Scale Vector AutoRegressive Models

The manuscript can be found here: [view paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619531)