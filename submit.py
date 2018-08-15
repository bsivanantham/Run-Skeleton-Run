import argparse
import numpy as np
from ast import literal_eval
from model import build_model_test, Agent
from environments import RunEnv2
from osim.http.client import Client

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy', dest='accuracy', action='store', default=5e-5, type=float)
    parser.add_argument('--modeldim', dest='modeldim', action='store', default='3D', choices=('3D', '2D'), type=str)
    parser.add_argument('--prosthetic', dest='prosthetic', action='store', default=1, type=int)
    parser.add_argument('--difficulty', dest='difficulty', action='store', default=0, type=int)
    parser.add_argument('--episodes', type=int, default=10, help="Number of test episodes.")
    parser.add_argument('--critic_layers', type=str, default='(64,64)', help="critic hidden layer sizes as tuple")
    parser.add_argument('--actor_layers', type=str, default='(64,64)', help="actor hidden layer sizes as tuple")
    parser.add_argument('--layer_norm', action='store_true', help="Use layer normalization.")
    parser.add_argument('--weights', type=str, default='weights.pkl', help='weights to load')
    parser.add_argument('--token', type=str, default='', help='token')
    args = parser.parse_args()
    args.modeldim = args.modeldim.upper()
    print('\n\ntoken  >>  ' + args.token + '\n\n')
    return args


def submit_agent(args, model_params):

    ##########################################################

    actor_fn, params_actor, params_crit = build_model_test(**model_params)
    weights = [p.get_value() for p in params_actor]
    actor = Agent(actor_fn, params_actor, params_crit)
    actor.set_actor_weights(weights)
    if args.weights is not None:
        actor.load(args.weights)

    env = RunEnv2(model=args.modeldim, prosthetic=args.prosthetic, difficulty=args.difficulty, skip_frame=3)

    # Settings
    remote_base = "http://grader.crowdai.org:1729"
    token = args.token
    client = Client(remote_base)

    # Create environment
    di = client.env_create(token, env_id="ProstheticsEnv")

    stat = []
    ep = 1
    ii = 0
    reward_sum = 0
    print('\n\n#################################################\n\n')
    while True:
        ii += 1
        proj = env.dict_to_vec(di)
        action = actor.act(proj)
        action += np.random.rand(len(action))/10.

        [di, reward, done, info] = client.env_step(action.tolist(), True)
        reward_sum += reward
        print('ep: ' + str(ep) + '  >>  step: ' + str(int(ii)) + '  >>  reward: ' + format(reward, '.2f') + '  \t' + str(int(reward_sum)) + '\t  >>  pelvis X Y Z: \t' + format(di['body_pos']['pelvis'][0], '.2f') + '\t' + format(di['body_pos']['pelvis'][1], '.2f') + '\t' + format(di['body_pos']['pelvis'][2], '.2f'))
        if done:
            print('\n\n#################################################\n\n')
            stat.append([ep, ii, reward_sum])
            di = client.env_reset()
            ep += 1
            ii = 0
            reward_sum = 0
            if not di:
                break
    for e in stat:
        print(e)
    print('\n\nclient.submit()\n\n')
    client.submit()
    ##########################################################
    print('\n\n#################################################\n\n')
    print('DONE\n\n')



def main():

    args = get_args()
    args.critic_layers = literal_eval(args.critic_layers)
    args.actor_layers = literal_eval(args.actor_layers)

    if args.prosthetic:
        num_actions = 19
    else:
        num_actions = 22

    env = RunEnv2(model=args.modeldim, prosthetic=args.prosthetic, difficulty=args.difficulty, skip_frame=3)
    env.change_model(args.modeldim, args.prosthetic, args.difficulty)
    env.spec.timestep_limit = 3000  # ndrw
    state = env.reset(seed=42, difficulty=0)
    d = env.get_state_desc()
    state_size = len(env.dict_to_vec(d))
    del env

    model_params = {
        'state_size': state_size,
        'num_act': num_actions,
        'gamma': 0,
        'actor_layers': args.actor_layers,
        'critic_layers': args.critic_layers,
        'actor_lr': 0,
        'critic_lr': 0,
        'layer_norm': args.layer_norm
    }

    submit_agent(args, model_params)


if __name__ == '__main__':
    main()

