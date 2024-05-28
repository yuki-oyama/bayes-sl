# Bayesian spatial logit model
Python code for the estimation of a hierarchical Bayesian logit model for spatial multivariate choice data.

## Paper
For more details, please see the paper

Oyama, Y., Murakami, D., Krueger, R. (forthcoming) [A hierarchical Bayesian logit model for spatial multivariate choice data](https://ssrn.com/abstract=4673510). Journal of Choice Modelling. 

If you find this code useful, please cite the paper:
```
@article{oyama2024bayes,
  title = {A hierarchical Bayesian logit model for spatial multivariate choice data},
  journal = {Journal of Choice Modelling},
  volume = {},
  pages = {},
  year = {2024},
  doi = {https://doi.org/10.1016/j.tra.2024.103998},
  url = {https://www.sciencedirect.com/science/article/pii/S0965856424000466},
  author = {Oyama, Yuki and Murakami, Daisuke and Krueger, Rico},
}
```

## Quick Start with Synthetic Data
**Estimate** a Bayesian spatial logit model with synthetic data, e.g., where the number of individual is 50 and that of spatial units is 200.

```
python run_synthetic.py --nInd 50 --nSpc 200
```
