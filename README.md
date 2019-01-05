# Black Box Optimization using Recurrent Neural Networks

## Abstract
We extend on the work of [Chen et al.](https://arxiv.org/abs/1611.03824), who introduce a new, learning-based, approach to global Black-Box optimization. They train a LSTM model, on functions sampled from a Gaussian process, to output an optimal sequence of sample points, i.e. a sequence of points that minimize those training functions. We verify the claims made by the authors, confirming that such a trained model is able to generalize to a wide variety of synthetic Black-Box functions, as well as to the real world problems of airfoil optimization and hyperparameter tuning of a SVM classifier. We show that the method performs comparably to state-of-the-art Black-Box optimization algorithms on these benchmarks, while outperforming them by a wide margin w.r.t. computation time. We thoroughly investigate the effects of different loss functions and training distributions on the methods performance and examine ways to improve it in the presence of prior knowledge.

![](https://i.imgur.com/4b0jRnB.jpg)

Full report and poster can be found at `./report`.



## Authors
- Pattarawat Chormai
- Felix Sattler
- Raphael Holca-Lammare(Supervisor)


## Development Setup
Please check `DEVELOPMENT.md`.
