import argparse

import torch



from procgen import ProcgenEnv



def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=-12345, help='random seed (default: -12345)') # Do not change the default value
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train')
    parser.add_argument(
        '--num-env-steps-tr',
        type=int,
        # default=10e6,
        help='number of environment steps to train')
    parser.add_argument(
        '--num-env-steps-pretrain',
        type=int,
        # default=10e4,
        help='number of environment steps to train')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='./logs',
        type=str,
        help='directory to save agent logs (default: .logs)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--cuda-num',
        type=str,
        default=False,
        help='provide gpu number')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--use-linear-lr-decay-rew',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--use-increase-decrease-lr-rew',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--use-increase-decrease-lr-pol',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--save-name',
        # action='store_true',
        type=str,
        # default=False,
        help='provide a name to be appended to the model')
    parser.add_argument(
        '--load-name',
        # action='store_true',
        type=str,
        # default=False,
        help='provide the save-name of the ranked trajs')
    parser.add_argument(
        '--main-function',
        type=str,
        choices=["train_GT_policy", "train_GAIL", "produce_ranked_trajs",  "produce_ranked_trajs_sparse","train_sparse_rank", 
        "train_baseline_sparse_rew", "train_Lipschitz_rew_from_ranks", "evaluate_policy_noisy_env", "visualize_policy"]
        )
    parser.add_argument(
        '--saved-models-name',
        type=str
        )
    parser.add_argument(
        '--state-normalize',
        action='store_true',
        default=False
        )
    parser.add_argument(
        '--rew-lr',
        type=float,
        # default=2.0e-4
        )   
    parser.add_argument(
        '--do-normalize',
        type=str,
        choices=['True', "False"]
        ) 
    parser.add_argument(
        '--rew-sign',
        type=str,
        choices=['free', "neg", "pos", "neg_sigmoid", "pos_sigmoid", "tanh"]
        ) 
    parser.add_argument(
        '--rew-mag',
        type=float,
        default=1.0
        ) 
    parser.add_argument(
        '--min-good-trajs',
        type=int,
        default=0
        ) 
    parser.add_argument(
        '--lam',
        type=float,
        default=1.0
        )  
    parser.add_argument(
        '--traj-length',
        type=int,
        )  
    parser.add_argument(
        '--num-rew-nets',
        type=int,
        )  
    parser.add_argument(
        '--rew-cntr',
        type=str,
        choices=['True', "False"]
        ) 
    parser.add_argument(
        '--run-type',
        type=str,
        choices=['main', "main_no_ranking", "main_opt_demos", "dev", "baseline", "pretrain_only"]
        ) 
    parser.add_argument(
        '--cntr-coeff',
        type=float
        ) 
    parser.add_argument(
        '--rew-coeff',
        type=float,
        default=1
        ) 
    parser.add_argument(
        '--pretrain',
        type=str,
        default="no",
        choices=["yes", "no", "load"])
    parser.add_argument(
        '--continue_',
        action='store_true',
        default=False,
        help='continue the simulation by reading the reward and policy from the drive')
    parser.add_argument(
        '--sparseness',
        type=float)
    parser.add_argument(
        '--num-states-update',
        type=int
        ) 
    parser.add_argument(
        '--num-perturbations',
        type=int
        ) 
    parser.add_argument(
        '--skip-rew-eval',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--priority-sampling',
        action='store_false',
        help='priority sampling for sampling pairs of trajectories from new trajectory buffer')
    parser.add_argument(
        '--num-opt-demo',
        type=int,
        # default=10e4,
        help='number of environment steps to train')   
    parser.add_argument(
        '--reg-beta',
        type=float,
        )  
    # arguments only for evaluate_reward_corr
    parser.add_argument(
        '--load-name-pol',
        type=str,
        )    
    parser.add_argument(
        '--load-name-rew',
        type=str,
        )
    parser.add_argument(
        '--load-rew-iter-idx',
        type=int,
        )  
    parser.add_argument(
        '--sparse-rew-type',
        type=str,
        choices=["units", "unitsV2", "steps", "episodic", "GT"]
        )  
    parser.add_argument(
        '--use-actions',
        action='store_true',
        default=False
        )
    parser.add_argument(
        '--shaping',
        action='store_true',
        default=False
        ) 
    parser.add_argument(
        '--reinit-rew',
        action='store_true',
        default=False
        ) 
    parser.add_argument(
        '--G-lip',
        action='store_true',
        default=False,
        help="If True, the generator of GAIL will be regularized to be lipschitz continuous"
        ) 
    parser.add_argument(
        '--D-lip',
        action='store_true',
        default=False,
        help="If True, the discriminator of GAIL will be regularized to be lipschitz continuous"
        ) 
    parser.add_argument(
        '--discounted-rew',
        action='store_true',
        default=False,
        help="If True, the reward learning will discount rewards"
        ) 
    parser.add_argument(
        '--lip-coeff',
        type=float
        ) 
    parser.add_argument(
        '--lip-norm',
        type=str,
        choices=["L_2","L_inf"],
        default=None
        ) 
    parser.add_argument(
        '--pert-radius',
        type=float,
        help="This is the maximum allowable norm infinity of the perturbation used for lipschitz training"
        ) 
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        ) 
    parser.add_argument(
        '--noisy-training',
        type=str,
        choices=["No","G","D","DG"],
        default="No",
        help="No means no noisy-training, G means only states visited by generator will be noisy, DG means both demonstrations and states visited by generator will be noisy"
        ) 
    parser.add_argument(
        '--train-noise',
        type=float,
        help="This is the maximum allowable norm infinity of the noise added to observations when the training environment is adversarial"
        )     
    parser.add_argument(
        '--adv-eval',
        action='store_true',
        default=False,
        ) 
    parser.add_argument(
        '--is-atari',
        action='store_true',
        default=False,
        )  
    parser.add_argument(
        '--sparse-cntr',
        action='store_true',
        default=False,
        ) 
    parser.add_argument(
        '--dont-remove-buffer',
        action='store_true',
        default=False,
        ) 
    # parser.add_argument(
    #     '--eval-noise',
    #     type=float,
    #     help="This is the maximum allowable norm infinity of the noise added to observations when the evaluation environment is adversarial"
    #     )  
    parser.add_argument(
        '--eval-name',
        type=str,
        help="save_name of the GAIL simulation where we want to evaluate"
        ) 
    parser.add_argument(
        '--num-overal-updates',
        type=int,
        default=0,
        help="if provided, it overwites the value in init_params"
        )  


    # # environment variables
    # parser.add_argument("--vision", choices=["agent", "human"], default="human")
    # parser.add_argument("--record-dir", help="directory to record movies to")
    # parser.add_argument("--distribution-mode", default="hard", help="which distribution mode to use for the level generation")
    # parser.add_argument("--level-seed", type=int, help="select an individual level to use")
    # parser.add_argument("--use-generated-assets", help="using autogenerated assets", choices=["yes","no"], default="no")
    # parser.add_argument("--obs-key", type=str, choices=["rgb", "state"])


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    kwargs = {}

    return args, kwargs

def get_init(myargs):

    run_type = myargs.run_type
    env_name = myargs.env_name
    is_atari = myargs.is_atari
    main_function = myargs.main_function
    init_params = {}

    if run_type in ['main_opt_demos']:
        # DEMONSTRATION HORIZON
        # init_params['demo_horizon'] = 1000
        # init_params['training_trajectory_skip'] = 5 # must be greater than one to separate training and evaluation demonstrations


        # NUMBER OF TRAINING SAMPLES AND BATCH SIZE
        # init_params['num_demos_batches_pretrain'] = 200
        # init_params['num_demos_batches_train'] = 20
        # init_params['num_reward_updates'] = 5 # number of reward updates after each policy update
        if myargs.reinit_rew:
            init_params['num_rew_training_batches'] = 60

        else:
            init_params['num_rew_training_batches'] = 10


        # NEW TRAJECTORY HYPER-PARAMETERS
        if is_atari:
            init_params["num_pol_updates_at_each_overal_update"] = 32

            init_params['size_of_new_trajs_list'] = 1000 # this is the maximum number of trajectories that will be stored
            # init_params['num_trajs_produced_each_iter'] = 20
            # init_params['num_trajs_first_time'] = 100
            init_params['produced_traj_length'] = 100

            # # SUBSAMPLING
            init_params['limited_buffer_length'] = True
            init_params['subsample_length'] = 50
            init_params['subsample_increment'] = 2 # must be greater than one to separate training and evaluation
            init_params['batch_size'] = 16 # batch size for reward updates


        else:
            init_params["num_pol_updates_at_each_overal_update"] = 64
            init_params['size_of_new_trajs_list'] = 15000 # this is the maximum number of trajectories that will be stored
            init_params['produced_traj_length'] = 500
            init_params['limited_buffer_length'] = True
            init_params['batch_size'] = 32 # batch size for reward updates
            
        # NUMBER OF UPDATES
        # init_params['num_first_policy_updates'] = 200 # number of policy iterations based on the pretrained reward
        # init_params['num_policy_updates'] = 5 # number of policy updates after each reward update
        init_params['save_reward_int'] = 25 # number of overal training iterations between saving successive rewards
        init_params['save_policy_int'] = 25 # number of overal training iterations between saving successive policies



        init_params['num_eval_pairs'] = 64


    elif run_type == 'dev':
        # DEMONSTRATION HORIZON
        init_params['demo_horizon'] = 200
        init_params['training_trajectory_skip'] = 2 # must be greater than one to separate training and evaluation demonstrations


        # NUMBER OF UPDATES
        init_params['num_overal_updates'] = 500 # iterations in the outermost loop
        init_params['num_first_policy_updates'] = 11 # number of policy iterations based on the pretrained reward
        init_params['num_policy_updates'] = 2 # number of policy updates after each reward update
        init_params['num_reward_updates'] = 2 # number of reward updates after each policy update
        init_params['save_reward_int'] = 20 # number of iterations between saving successive rewards
        init_params['save_policy_int'] = 20 # number of iterations between saving successive rewards

        # NUMBER OF TRAINING SAMPLES AND BATCK SIZE
        init_params['num_demos_batches_pretrain'] = 5
        init_params['num_demos_batches_train'] = 5
        init_params['num_policyTraj_batches_train'] = 5
        init_params['batch_size'] = 16 # batch size for reward updates

        # NEW TRAJECTORY HYPER-PARAMETERS
        init_params['size_of_new_trajs_deque'] = 1000
        init_params['num_trajs_produced_each_iter'] = 10
        init_params['produced_traj_length'] = 50 


    if main_function in ['produce_ranked_trajs_sparse']:
        init_params['produced_traj_length'] = 1000

    return init_params

def get_init_baseline(myargs):
    init_params = {}
    is_atari = myargs.is_atari
    if is_atari:
        init_params['num_overal_updates'] = 2500 # iterations in the outermost loop
    else:
        init_params['num_overal_updates'] = 500 # iterations in the outermost loop


    init_params['save_pol_int'] = 500

    return init_params

