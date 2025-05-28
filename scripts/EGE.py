import argparse
import numpy as np
import Algorithm
import Bandits
from tqdm import tqdm
import time


def parse_args():
    """
    Specifies command line arguments for the program.
    """
    parser = argparse.ArgumentParser(description='Best arm identification')

    parser.add_argument('--seed', default=1, type=int,
                        help='Seed for random number generators')
    # default best-arm options
    parser.add_argument('--n', default=10, type=int,
                        help='number of total arms')
    parser.add_argument('--delta', default=0.1, type=float,
                        help='1 - target confidence')
    parser.add_argument('--method', default='TrackandStop',

                        help='method')
    parser.add_argument('--num_sim', default=1000, type=int,
                        help='number of total simulation')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    np.set_printoptions(threshold=np.inf)
    num_arms = args.n
    delta = args.delta
    #epsilon = 1/np.log(np.log(1/delta))
    epsilon=0.01
    print('epsilon:',epsilon)
    alpha = 1.001
    L1 = int(np.ceil(60*np.sqrt(np.log(1/delta))))+40
    #L1 = int(np.ceil(23 * np.sqrt(np.log(1 / delta))))-8
    #L1= 330
    print('L1:',L1)
    #L2 = int(np.ceil(c*np.log(1/delta)*np.log(np.log(1/delta))/num_arms))
    #L2 = int(np.ceil(c * np.log(1 / delta) * np.log(np.log(1 / delta)) / num_arms))
    L2= int(np.ceil(10*np.log(1/delta)))+40
    #L2 = int(np.ceil(3 * np.log(1 / delta)))+10
    #L2= 250
    print('L2:',L2)
    #L3 = int(np.ceil(12*np.power(np.log(1/delta),2)))
    L3= 180
    print('L3:',L3)
    num_error=0
    # uniform
    arm_means_uniform = np.random.uniform(0.2, 0.4, num_arms)
    arm_means_uniform[0] = 0.5
    print('arm_means_uniform:', arm_means_uniform)
    # normal
    arm_means_normal = np.random.normal(0.2, 0.2, num_arms)
    for i in range(num_arms):
        mean = arm_means_normal[i]
        while mean > 0.4 or mean < 0:
            mean = np.random.normal(0.2, 0.2)
        arm_means_normal[i] = mean
    arm_means_normal[0] = 0.6
    print('arm_means_normal:', arm_means_normal)
    delta_list=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
    for k in range(len(delta_list)):
        delta=delta_list[k]
        time_list = []
        sample_list = []
        batch_list=[]
        for i in tqdm(range(args.num_sim)):
            start_time = time.time()
            # these parameters can also be set individually for each arm

            sim = Bandits.Simulator(num_arms=num_arms, arm_means=arm_means_uniform)
            total_sample, mu, best_arm, batch = Algorithm.ExponentialGapElimination(delta, sim)
            batch_list.append(batch)
            if best_arm!=np.argmax(arm_means_uniform)+1:
                num_error+=1

            sample_list.append(total_sample)
            end_time = time.time()
            time_list.append(end_time-start_time)
        print('error_rate:',num_error/args.num_sim)
        print('Runtime Mean:', np.mean(time_list))
        print('Runtime Std:', np.std(time_list))
        print('Sample Mean:', np.mean(sample_list))
        print('Sample Std:', np.std(sample_list))
        print('Batch Mean:', np.mean(batch_list))
        print('Batch Std:', np.std(batch_list))