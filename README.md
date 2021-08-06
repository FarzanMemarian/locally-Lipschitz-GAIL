# Run the code
To run the code (for Lipschitz regularized discriminator), go to the main directory and run some command like the following

python main.py --main-function "train_GAIL" --env-name "Hopper-v2" --D-lip --lip-norm "L_2" --gail --algo ppo --use-gae --lr 3.0e-4 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --num-processes 8 --num-steps 2048 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.0  --save-interval 10 --num-env-steps 15000000 --use-proper-time-limits --eval-interval 20 --seed 123



# base code

This code is developed based on the following implementation of the PPO algorithm.
    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
    }

