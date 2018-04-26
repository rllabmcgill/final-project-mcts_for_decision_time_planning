# Probabilistic MCTS for navigating to a goal in a dangerous environment

An agent (green) uses MCTS to find trajectories through a busy road with random initialization and goal points (yellow cross).

Example code:
[roadway_modelpy](https://github.com/rllabmcgill/MCTS_function_approximation/blob/master/trajectories/examples/roadway_model.py)

This repo will not be updated after May 2018, please refer to the living repo here:
[https://github.com/johannah/trajectories](https://github.com/johannah/trajectories)

We learned a model of the environment using a vqvae and perform one-step ahead rollouts using this model. 

![alt_text](https://github.com/rllabmcgill/MCTS_function_approximation/blob/master/counterpoint/true_step_seed_930_vqvae.gif)

![alt_text](https://github.com/rllabmcgill/MCTS_function_approximation/blob/master/counterpoint/playout_step_seed_930_vqvae.gif)


We learned a model of the environment using a vae and perform one-step ahead rollouts using this model. 

![alt_text](https://github.com/rllabmcgill/MCTS_function_approximation/blob/master/counterpoint/true_step_seed_930_vae.gif)

![alt_text](https://github.com/rllabmcgill/MCTS_function_approximation/blob/master/counterpoint/playout_step_seed_930_vae.gif)


## Authors:
Johanna Hansen | McGill Student ID: 260704014 | email: johanna.hansen@mail.mcgill.ca
