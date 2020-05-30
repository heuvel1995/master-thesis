import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import random
import string
import warnings
from collections import deque
from scipy.optimize import fsolve
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
from scipy.stats import ttest_ind, binom, norm
import time
import pickle
import dill
import pandas as pd
import os
from scipy.stats import ttest_ind
import copy
from math import ceil
import math
from joblib import Parallel, delayed
import builtins

warnings.filterwarnings("ignore")
pd.options.display.max_rows = 999


def phi(n_pulls, delta):
    """ Calculates phi to use in confidence interval calculations.

    :param n_pulls int: number of pulls taken from this arm.
    :param delta float: confidence level.

    :return: phi float: phi.
    """
    item_1 = np.log(1 / delta)
    item_2 = 3 * np.log(np.log(1 / delta))
    item_3 = (3 / 2) * np.log(np.log(np.e * n_pulls))
    return np.sqrt((item_1 + item_2 + item_3) / n_pulls)


def create_folder(directory):
    """ Creates a folder in a specified directory.

    :param directory (str): directory to create.

    :return: None (a directory is created).
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory {}'.format(directory))


class MetaExperiment:
    def __init__(self,
                 policies: list,
                 k_list: list,
                 alphas: list,
                 initial_pulls_list: list,
                 n_users: int,
                 n_variants: int,
                 max_pulls: int,
                 p_range: tuple,
                 path: str,
                 pulls_per_round: int = 1,
                 p_increase: float = 0.0,
                 bootstrap_settings: dict = {'iterations': 500,
                                             'bootstrap_method': 'online'},
                 write_queue = False,
                 read_queue = False,
                 queue_path: str = None,
                 history = False):
        """
        In each meta-experiment, we test one unique combination of a policy and alpha.

        :param policies list: all policies we want to loop over.
        :param k_list list: number of arms we want to test.
        :param alphas list: alphas we want to test.
        :param initial_pulls list: list with number of initial pulls to take per arm in an experiment.
        :param n_users int: number of users allowed per simulation.
        :param n_variants int: number of variants to use in total (length of the variant_queue)
        :param max_pull int: maximum number of pulls per experiment.
        :param p_range tuple(float, float): range from which to start sampling p (success probability).
        :param path str: path to save the experiment.
        :param pulls_per_round int: number of pulls to take per round of an experiment (Assigner, Puller, Evaluator,
        Terminator).
        :param p_increase float: absolute increase with which the p_range is increased over
        ceil(n_users/initial_pulls) iterations.
        :param bootstrap_settings dict: specifies the number of bootstrap iterations and method.
        :param write_queue bool: if True, write the user_queues and variant_queues to a pickle.
        :param read_queue bool: if True, read the user_queues and variant_queues from a pickle.
        :param queue_path str: path to read or write the queues.
        :param history bool: if True, keep track of the history of the experiment (which arms were pulled in which sequence)

        """

        # Test variables, input as list
        self.policies = policies
        self.k_list = k_list
        self.alphas = alphas
        self.initial_pulls_list = initial_pulls_list
        self.n_users = n_users
        self.n_variants = n_variants
        self.max_pulls = max_pulls
        self.p_range = p_range
        self.path = path

        # Default arguments
        self.pulls_per_round = pulls_per_round
        self.p_increase = p_increase
        self.bootstrap_settings = bootstrap_settings
        self.write_queue = write_queue
        self.read_queue = read_queue
        self.queue_path = queue_path
        self.history = history

        # Create an empty list to store all the simulations of the meta-experiment.
        self.simulations = []

    def run(self, name):
        """ Runs a MetaExperiment.

        :param name (str): name which is used to save the MetaExperiment.

        :return: None (the MetaExperiment is saved in self.path/name).
        """

        # Load queues if applicable
        if self.read_queue:
            queues = pickle.load(open("{}/{}.p".format(self.queue_path,name),'rb'))
            self.user_queue = queues['user_queue']
            self.variant_queue = queues['variant_queue']

        # Create queues if applicable
        else:
            # Create user_queue
            self.user_queue = deque(np.random.uniform(0, 1, int(self.n_users)))

            # Create variant_queue
            self.variant_queue = deque()
            for _ in range(self.n_variants):
                self.variant_queue.append(random.uniform(self.p_range[0], self.p_range[1]))
                self.p_range = (self.p_range[0] + self.p_increase / self.n_variants,
                                self.p_range[1] + self.p_increase / self.n_variants)

        # Write queues if applicable
        if self.write_queue:
            pickle.dump({'user_queue':self.user_queue,
                         'variant_queue':self.variant_queue},open("{}/{}.p".format(self.queue_path,name),'wb'))
        else:
            pass

        # Use a counter to keep track of the progress
        progress_counter = 1

        # Calculate how many tests we have to run, useful for tracking progress
        length = len(self.alphas) * len(self.policies) * len(self.k_list) * len(self.initial_pulls_list)

        # Run a GridSearch over all combinations of policies, k_list and alphas
        for policy in self.policies:
            for k in self.k_list:
                for alpha in self.alphas:
                    if policy in ['A/B testing', 'LUCB']:
                        start = time.time()
                        print('Starting test {} of {}'.format(progress_counter, length))

                        # Set up the simulation
                        simulation = Simulation(self, policy, k, alpha, self.initial_pulls_list[0])

                        # Run the simulation
                        simulation.run()

                        # Calculate the runtime
                        simulation.runtime = time.time() - start

                        # Append the entire Simulation object to a self.simulations
                        self.simulations.append(simulation)

                        # Update the progress
                        progress_counter += 1
                    else:

                        # Loop over all possible number of initial_pulls.
                        for initial_pulls in self.initial_pulls_list:
                            start = time.time()
                            print('Starting test {} of {}'.format(progress_counter, length))

                            # Set up the simulation
                            simulation = Simulation(self, policy, k, alpha, initial_pulls)

                            # Run the simulation
                            simulation.run()

                            # Calculate the runtime
                            simulation.runtime = time.time() - start

                            # Append the entire Simulation object to a self.simulations
                            self.simulations.append(simulation)

                            # Update the progress
                            progress_counter += 1

        # Dump the result of the meta-experiment to a pickle
        dill.dump(self.evaluate(summary=False), open("{}\\result\\{}.p".format(self.path, name), 'wb'))

        # Delete some excess information to reduce the file size
        for simulation in self.simulations:
            for experiment in simulation.experiments:
                for arm in experiment.arms:
                    del arm.rewards

        # Dump the meta-experiment to a pickle
        dill.dump(self, open("{}\\{}.p".format(self.path,name),'wb'))


    def evaluate(self,summary=True):
        """ Evaluates a MetaExperiment.

        :param self MetaExperiment: MetaExperiment object, which holds a list of Simulations.
        :param summary bool: if summary is True, group the DataFrame by policy and alpha.

        :return: pandas DataFrame, either summarized or not.
        """

        columns = ['experiment_nr', 'policy', 'k', 'alpha', 'initial_pulls'] + \
                  [i for i in range(self.k_list[0])] + \
                  ['best_arm', 'returned_arm', 'label', 'regret', 'net_regret', 'total_reward', 'sample_complexity']

        # List to store the result of every experiment of every simulation
        result = []
        for simulation_nr, simulation in enumerate(self.simulations):
            for experiment_nr, experiment in enumerate(simulation.experiments):
                if not experiment.valid_experiment:
                    pass
                else:
                    experiment.review()
                    experiment_result = [experiment_nr]
                    experiment_result.extend([simulation.policy, simulation.k, simulation.alpha, simulation.initial_pulls])
                    experiment_result.extend([(round(arm.true_mean,3), round(arm.mean,3)) for arm in experiment.arms])
                    experiment_result.extend([experiment.true_best_arm.identifier, experiment.returned_arm.identifier,
                                              experiment.label, experiment.regret, experiment.net_regret,
                                              np.sum([np.sum(arm.rewards) for arm in experiment.arms]),experiment.total_pulls])
                    result.append(experiment_result)

        # Transform the results into a nice DataFrame
        df = pd.DataFrame(columns=columns, data=result)
        for label in ['TP', 'TN', 'FP', 'FN']:
            df[label] = df['label'].apply(lambda x: 1 if x == label else 0)
        df = df.drop([0, 1, 2, 3,'best_arm', 'returned_arm'], axis=1)

        if not summary:
            return df
        else: # Summarize the DataFrame by policy, alpha, k, and initial_pulls
            df_grouped = df.groupby(['policy', 'alpha', 'k', 'initial_pulls']).sum()
            df_grouped = df_grouped[['regret', 'net_regret', 'total_reward', 'sample_complexity', 'TP', 'TN', 'FP', 'FN']]
            df_grouped['MCC'] = df_grouped.apply(lambda row: ((row['TP']*row['TN']) - (row['FP']*row['FN']))/
                                                             np.sqrt((row['TP'] + row['FP'])*(row['TP'] + row['FN'])
                                                             *(row['TN'] + row['FP'])*(row['TN'] + row['FN'])), axis=1)
            df_grouped['precision'] = df_grouped['TP'] / (df_grouped['TP'] + df_grouped['FP'])
            df_grouped['recall'] = df_grouped['TP'] / (df_grouped['TP'] + df_grouped['FN'])
            df_grouped = df_grouped.reset_index()
            return df_grouped

class Simulation:
    def __init__(self,
                 meta_experiment,
                 policy: str,
                 k: int,
                 alpha: float,
                 initial_pulls: int):
        """
        In each Simulation, we run multiple experiments, using a specified policy, K, alpha and initial_pulls

        :param meta_experiment obj: MetaExperiment object which holds important parameters for this simulation.
        :param policy str: the policy to use in this simulation, possible values: "A/B testing", "Boostrapped LUCB",
        "LUCB", "TTTS".
        :param k int: number of arms to use per experiment
        :param alpha float: significance level
        :param initial_pulls int: number of initial_pulls to employ

        :return Creates a Simulation object.
        """
        self.meta_experiment = meta_experiment
        self.policy = policy
        self.k = k
        self.alpha = alpha
        self.initial_pulls = initial_pulls

        # Create copies of the queues in the meta-experiment, such that they can be re-used in other meta-experiments.
        self.user_queue = copy.deepcopy(self.meta_experiment.user_queue)
        self.variant_queue = copy.deepcopy(self.meta_experiment.variant_queue)
        self.experiments = []

    def run(self):
        """
        Runs the simulation. Takes no parameters, all parameters are stored in the object.

        :return self
        """

        # For the first experiment, get K arms from the variant_queue.
        arms = []
        for index in range(self.k):
            arms.append(Arm(identifier=index,
                            alpha=self.alpha,
                            p=self.variant_queue.popleft(),
                            policy=self.policy,
                            k=self.k,
                            bootstrap_settings=self.meta_experiment.bootstrap_settings))

        # Set up the first experiment
        experiment = Experiment(self, arms)

        start = time.time()

        # Run the first experiment
        experiment.run()
        end = time.time()

        self.experiments.append(experiment)

        # While there are still users left, continue to run experiments
        while len(self.user_queue) > 0:

            # The returned arm of the previous experiment is assigned to be the control arm in the next experiment.
            arms = [Arm(identifier=0,
                        alpha=self.alpha,
                        p=experiment.returned_arm.p,
                        policy=self.policy,
                        k=self.k,
                        bootstrap_settings=self.meta_experiment.bootstrap_settings)]

            # If there are still enough variants left to evaluate, append K-1 arms to the arms list
            if len(self.variant_queue) >= self.k-1:
                for index in range(1, self.k):
                    arms.append(Arm(identifier=index,
                                    alpha=self.alpha,
                                    p=self.variant_queue.popleft(),
                                    policy=self.policy,
                                    k=self.k,
                                    bootstrap_settings=self.meta_experiment.bootstrap_settings))

            # If there's not enough arms for a new experiment, only pull the control arm for the remainig pulls
            else:
                pass

            # Set up experiment
            experiment = Experiment(self, arms)

            start = time.time()

            # Run experiment
            experiment.run()
            end = time.time()

            # Add the experiment to self.experiments, which holds a list of all experiments in this simulation
            self.experiments.append(experiment)

        return self

class Experiment:
    def __init__(self,
                 simulation,
                 arms: list):
        """
        In each Experiment, we test K arms using a specified policy and alpha.

        :param simulation obj: Simulation object which holds important parameters for this simulation.
        :param arms list: list of Arm objects, the arms to be evaluated in this experiment.

        :return Creates a Experiment object.
        """
        # Set up the arms
        self.simulation = simulation
        self.arms = arms

        # Other arguments
        self.control_arm = self.arms[0]
        self.history = []
        self.total_pulls = 0
        self.returned_arm = self.control_arm
        self.regret = 0
        self.should_stop = False
        self.true_best_arm = self.arms[np.argmax([arm.true_mean for arm in self.arms])]
        self.valid_experiment = True
        self.max_pulls = self.simulation.meta_experiment.max_pulls
        self.pulls_per_round = self.simulation.meta_experiment.pulls_per_round

    def run(self):
        """
        Runs the experiment.  Takes no parameters, all parameters are stored in the object.

        :return self (Experiment object).
        """
        # Check how many users are left in this simulation
        users_left = len(self.simulation.user_queue)

        # If we are only testing one arm, the variant_queue is empty.
        # Assign all remaining users to this arm.
        if len(self.arms) == 1:
            assignment_queue = deque([self.control_arm for _ in range(users_left)])
            self.take_pulls(assignment_queue)
            self.valid_experiment = False

        # If there are too few users left to conduct a full experiment, only pull the control arm.
        elif users_left < self.max_pulls:
            assignment_queue = deque([self.control_arm for _ in range(users_left)])
            self.take_pulls(assignment_queue)
            self.regret = self.calculate_regret()
            self.valid_experiment = False

        # Otherwise, run a true experiment.
        # Note this is what is happens of the time.
        else:
            while not self.should_stop:
                # Call the Assigner to create the assignment_queue
                assignment_queue = self.assigner()

                # Create a set of the arms that were pulled such that we don't run the Evaluator on arms that did not change.
                pulled_arms = set(assignment_queue)

                # Call the puller (note it is called slightly different than in the paper)
                self.take_pulls(assignment_queue)

                # Call the Evaluator to evaluate the arms which have been pulled
                self.evaluator(pulled_arms)

                # Call the Terminator to see if we can terminate the experiment. This updates self.should_stop.
                self.terminator()

            # Run the Decider to decide on the winning arm
            self.decider()

        return self

    def create_assignment_queue(self, selected_arms, n_pulls):
        """
        Helper function to create an assignment_queue.

        :param selected_arms: set of selected arms, in which each element is an Arm object
        :param n_pulls int: number of pulls to divide amongst these Arms

        :return assignment_queue
        """
        pulls_left = self.max_pulls - self.total_pulls
        n_pulls = np.min([pulls_left, n_pulls])

        pulls_per_arm = math.floor(n_pulls / len(selected_arms))
        assignment_list = [arm for _ in range(pulls_per_arm) for arm in selected_arms]
        while len(assignment_list) < n_pulls:
            assignment_list.append(selected_arms[0])
        return deque(assignment_list)

    def assigner(self):
        """
        Assigner which is called to create the assignment_queue. The behaviour of the Assigner is dependent on the
        policy. ALl parameter reside in self (Experiment object)

        :return assignment_queue
        """
        # For the logic supporting this policy, see Section 3.2.1.
        if self.simulation.policy == 'LUCB':
            # If this is the first round, assign each arm once.
            if self.total_pulls == 0:
                return self.create_assignment_queue(self.arms, self.pulls_per_round)
            else:
                # Find the best_arm (arm with the highest mean)
                best_arm_index = np.argmax([arm.mean for arm in self.arms])
                best_arm = self.arms[best_arm_index]

                # Find the best contender (arm with the highest UCB that is not the best arm)
                ucbs = [arm.UCB for arm in self.arms]
                ucbs[best_arm_index] = 0
                best_contender_index = np.argmax([ucbs])
                best_contender = self.arms[best_contender_index]
                selected_arms = [best_arm, best_contender]

                # Return the assignment_queue
                return self.create_assignment_queue(selected_arms, self.pulls_per_round)

        # For the logic supporting this policy, see Section 3.2.2.
        elif self.simulation.policy == 'Bootstrapped LUCB':
            # If the initial_pulls have not yet been taken, take the initial_pulls
            if self.total_pulls == 0:
                return self.create_assignment_queue(self.arms, self.simulation.initial_pulls*len(self.arms))

            else:
                # Find the best_arm (arm with the highest mean)
                best_arm_index = np.argmax([arm.mean for arm in self.arms])
                best_arm = self.arms[best_arm_index]

                # Find the best contender (arm with the highest UCB that is not the best arm)
                ucbs = [arm.UCB for arm in self.arms]
                ucbs[best_arm_index] = 0
                best_contender_index = np.argmax([ucbs])
                best_contender = self.arms[best_contender_index]
                selected_arms = [best_arm, best_contender]

                # If the best arm does not yet beat the control arm and it is not yet selected, also select the control arm
                if self.simulation.policy == 'Bootstrapped LUCB':
                    if best_arm.LCB < self.control_arm.UCB and self.control_arm not in selected_arms:
                        selected_arms.append(self.control_arm)

                # Return the assignment_queue
                return self.create_assignment_queue(selected_arms, self.pulls_per_round)

        # For the logic supporting this policy, see Section 3.1
        elif self.simulation.policy == 'A/B testing':
            return self.create_assignment_queue(self.arms, self.max_pulls)

        # For the logic supporting this policy, see Section 3.2.3.
        # This code is inspired by Shaw Lu.
        # https://towardsdatascience.com/beyond-a-b-testing-multi-armed-bandit-experiments-1493f709f804
        elif self.simulation.policy == 'TTTS':
            # If the initial_pulls have not yet been taken, take the initial_pulls
            if self.total_pulls == 0:
                return self.create_assignment_queue(self.arms, self.simulation.initial_pulls*len(self.arms))

            else:
                # Find the best arm (arm with the highest sample)
                best_arm = self.arms[np.argmax([arm.sample() for arm in self.arms])]

                # Find teh best contender (arm with the highest sample that is not the best arm)
                best_contender = best_arm
                while best_arm == best_contender:
                    best_contender = self.arms[np.argmax([arm.sample() for arm in self.arms])]
        
                selected_arms = [best_arm, best_contender]
                # Return the assignment_queue
                return self.create_assignment_queue(selected_arms, self.pulls_per_round)

        else:
            print("Policy not found")

    def take_pulls(self, assignment_queue):
        """
        Pulls each arm in the assignment_queue until it is empty and store the result in the corresponding arm.

        :return None (results are stored in the arms and the Experiment object)
        """
        while len(assignment_queue) > 0:
            arm = assignment_queue.popleft()
            arm.pull(self.simulation.user_queue.popleft())
            self.total_pulls += 1

            if self.simulation.meta_experiment.history:
                self.history.append(arm)

    def evaluator(self, pulled_arms):
        """
        Evaluate all of the arms which have been pulled in this round.

        :return None (results are stored in the arms and the Experiment object)
        """
        if self.simulation.policy == 'LUCB':
            for arm in self.arms:
                arm.rewards.extend(arm.rewards_to_process)
                arm.rewards_to_process = []
                arm.update_mean()
                arm.update_confidence_bound(method = 'LUCB',
                                            online = False,
                                            iterations = self.simulation.meta_experiment.bootstrap_settings['iterations'],
                                            k = self.simulation.k)

                arm.mean_history.append((self.total_pulls, arm.mean))
                arm.UCB_history.append((self.total_pulls, arm.UCB))
                arm.LCB_history.append((self.total_pulls, arm.LCB))

        elif self.simulation.policy == 'Bootstrapped LUCB':
            for arm in pulled_arms:
                arm.update_confidence_bound(method = 'bootstrap',
                                            online = self.simulation.meta_experiment.bootstrap_settings['online'],
                                            iterations = self.simulation.meta_experiment.bootstrap_settings['iterations'],
                                            k = self.simulation.k)

                arm.mean_history.append((self.total_pulls, arm.mean))
                arm.UCB_history.append((self.total_pulls, arm.UCB))
                arm.LCB_history.append((self.total_pulls, arm.LCB))

        elif self.simulation.policy == 'A/B testing':
            for arm in self.arms:
                arm.rewards.extend(arm.rewards_to_process)
                arm.rewards_to_process = []
                arm.update_mean()

            # For every arm that is not the control arm, calculate the one-sided p-value with respect to the control arm.
            self.control_arm.p_value = 1
            for arm in self.arms[1:]:
                if arm.mean > self.control_arm.mean:
                    arm.p_value = (ttest_ind(self.control_arm.rewards, arm.rewards).pvalue)/2
                else:
                    arm.p_value = 1

        elif self.simulation.policy == 'TTTS':
            for arm in self.arms:
                # Update the beta-distribution of all the arms
                arm.a += int(np.sum(arm.rewards_to_process))
                arm.b += int(len(arm.rewards_to_process) - np.sum(arm.rewards_to_process))
                arm.rewards.extend(arm.rewards_to_process)
                arm.mean = np.mean(arm.rewards)
                arm.rewards_to_process = []

    def terminator(self):
        """
        Check is we can terminate the experiment.

        :return None (updates self.should_stop)
        """
        # For the LUCB policies, if one of the arms beats all other arms, terminate the experiment.
        # This happens when the lower confidence bound of the best arm is better than the upper confidence bound of all other arms.
        if self.simulation.policy == 'LUCB' or self.simulation.policy == 'Bootstrapped LUCB':
            if self.total_pulls >= self.max_pulls:
                self.should_stop = True
            else:
                best_arm = self.arms[np.argmax([arm.mean for arm in self.arms])]

                if all(best_arm.LCB > arm.UCB for arm in self.arms if arm.identifier != best_arm.identifier):
                    self.should_stop = True
                else:
                    pass

        elif self.simulation.policy == 'A/B testing':
            self.should_stop = True

        elif self.simulation.policy == 'TTTS':
            if self.total_pulls >= self.max_pulls:
                self.should_stop = True

            # Run a monte-carlo simulation with all arms, stop if 1 - alpha of  the  samples  in  have  a  PVR  of less
            # Than 1% of the current best arm’s value
            # This code is inspired by Shaw Lu.
            # https://towardsdatascience.com/beyond-a-b-testing-multi-armed-bandit-experiments-1493f709f804
            else:
                # record current estimates of each arm being winner
                mc, p_winner = self.monte_carlo_simulation(self.arms)

                best_arm_index = np.argmax(p_winner)
                values_remaining = (mc.max(axis=1) - mc[:, best_arm_index]) / mc[:, best_arm_index]
                pctile = np.percentile(values_remaining, q=100 * (1 - self.simulation.alpha))

                if pctile < self.simulation.alpha * self.arms[best_arm_index].mean:
                    self.should_stop = True
                else:
                    self.should_stop = False

    def decider(self):
        """
        Find the winning arm of an experiment. See Section 3.2 for the policies.

        :return None (self.returned_arm is updated)
        """
        if self.simulation.policy == 'LUCB' or self.simulation.policy == 'Bootstrapped LUCB':
            beating_arms = [arm for arm in self.arms if arm.LCB > self.control_arm.UCB]
            if len(beating_arms) == 0:
                self.returned_arm = self.control_arm
            else:
                self.returned_arm = beating_arms[np.argmax([arm.mean for arm in beating_arms])]

        elif self.simulation.policy == 'A/B testing':
            beating_arms = [arm for arm in self.arms if arm.p_value < self.simulation.alpha]
            if len(beating_arms) == 0:
                self.returned_arm = self.control_arm
            else:
                self.returned_arm = beating_arms[np.argmin([arm.p_value for arm in beating_arms])]

        elif self.simulation.policy == 'TTTS':
            best_arm_index = np.argmax([arm.mean for arm in self.arms])

            mc, p_winner = self.monte_carlo_simulation([self.arms[0], self.arms[best_arm_index]])

            values_remaining = (mc.max(axis=1) - mc[:, 1]) / mc[:, 1]
            pctile = np.percentile(values_remaining, q=100 * (1 - self.simulation.alpha))

            if pctile < self.simulation.alpha * self.arms[best_arm_index].mean:
                self.returned_arm = self.arms[best_arm_index]
            else:
                self.returned_arm = self.control_arm


    def monte_carlo_simulation(self, arms, draw=500):
        """
        Monte Carlo simulation. Each arm's reward follows a beta distribution.
        This code is inspired by Shaw Lu.
        https://towardsdatascience.com/beyond-a-b-testing-multi-armed-bandit-experiments-1493f709f804

        :param list[Arm]: list of Arm objects.
        :param draw int: number of draws in Monte Carlo simulation.

        :returns mc np.matrix: Monte Carlo matrix of dimension (draw, k).
        :returns p_winner list[float]: probability of each arm being the winner.
        """
        # Monte Carlo sampling
        alphas = [arm.a for arm in arms]
        betas = [arm.b for arm in arms]
        mc = np.matrix(np.random.beta(alphas, betas, size=[draw, len(arms)]))

        # count frequency of each arm being winner
        counts = [0 for _ in arms]
        winner_idxs = np.asarray(mc.argmax(axis=1)).reshape(draw,)
        for idx in winner_idxs:
            counts[idx] += 1

        # divide by draw to approximate probability distribution
        p_winner = [count / draw for count in counts]
        return mc, p_winner

    def calculate_regret(self):
        """
        Helper function to calculate the regret of the experiment.

        :returns regret float: cumulative regret of the experiment.
        """
        total_reward = np.sum([np.sum(arm.rewards) for arm in self.arms])
        potential_reward = np.max([arm.true_mean for arm in self.arms]) * self.total_pulls
        return potential_reward - total_reward

    def review(self):
        """
        Reviews the performance of the experiment, including the label (TP, TN, FP, FN) and regret.

        :returns None, relevant elements are stored in self (Experiment object).
        """
        if not self.valid_experiment:
            return None

        # There is a better arm and we correctly conclude so. Note that here there is a off-chance that the returned
        # arm is not the true best arm, but still the null hypothesis is correctly rejected
        if (self.true_best_arm != self.control_arm) & (self.returned_arm != self.control_arm):
            self.label = 'TP'
        # There is no better arm and we correctly concluded this
        elif (self.true_best_arm == self.control_arm) & (self.returned_arm == self.control_arm):
            self.label = 'TN'
        # There is no better arm, but we erroneously say there is
        elif (self.true_best_arm == self.control_arm) & (self.returned_arm != self.control_arm):
            self.label = 'FP'
        # There is a better arm but we fail to identify it
        elif (self.true_best_arm != self.control_arm) & (self.returned_arm == self.control_arm):
            self.label = 'FN'

        if self.true_best_arm == self.control_arm:
            self.uplift = 0
        else:
            self.uplift = round((self.true_best_arm.mean - self.control_arm.mean)/self.control_arm.mean * 100,2)

        total_reward = np.sum([np.sum(arm.rewards) for arm in self.arms])
        potential_reward = np.max([arm.true_mean for arm in self.arms]) * self.total_pulls
        self.regret = potential_reward - total_reward

        net_total_reward = np.sum([np.sum(arm.rewards[self.simulation.initial_pulls:]) for arm in self.arms])
        net_potential_reward = np.max([arm.true_mean for arm in self.arms]) *\
                               (self.total_pulls - self.simulation.initial_pulls * len(self.arms))
        self.net_regret = net_potential_reward - net_total_reward


class Arm:
    def __init__(self,
                 identifier,
                 alpha,
                 p,
                 policy,
                 k,
                 bootstrap_settings):
        """
        Arm object, in which relevant elements for this Arm can be stored.

        :param alpha float: significance level used to evaluate this arm.
        :param p float: success probability
        :param policy str: policy of the experiment in which this arm is used.
        :param k: number of arms in the experiment, relevant for LUCB.
        :param: bootstrap_settings, see MetaExperiment.

        :return Creates an Arm object.
        """
        self.identifier = identifier
        self.alpha = alpha
        self.p = p
        self.policy = policy
        self.k = k
        self.bootstrap_settings = bootstrap_settings

        self.mean = np.nan
        self.UCB = 0
        self.LCB = 0
        self.mean_history = []
        self.UCB_history = []
        self.LCB_history = []

        # Rewards to process are processed in the Evaluator.
        self.rewards = []
        self.rewards_to_process = []

        # Since we only run binomial experiment, true_mean = p
        self.true_mean = self.p

        # Beta distribution parameters for TTTS
        self.a = 1
        self.b = 1

    def pull(self, quantile):
        """
        Helper function to pull the arm. For Binomial expeirment, the logic is the same as Inverse Transform Sampling,
        but this is slightly more efficient than scpipy.

        :param quantile float: the quantile represents some user's popensity to buy. If it is higher, reward lis more
        likely to be 1.

        :return: None
        """
        if quantile < self.p:
            self.rewards_to_process.append(1)
        else:
            self.rewards_to_process.append(0)

    def update_mean(self):
        self.mean = np.mean(self.rewards)

    def update_confidence_bound(self, method, online, iterations, k=None):
        """
        Update the confidence bounds for arms with LUCB or Bootstrapped LUCB policy.

        :param method str: defines how the confidence bounds should be updated.
        :param online bool: if online, update the confidence bounds in an online fashion (only relevant for Bootstrapped LUCB)
        :param iterations int: number of bootstrap iterations
        :param k int: number of arms in the experiment.

        :return: None
        """
        if method == 'LUCB':
            # See Section 3.2.1 of the paper.
            self.LCB = self.mean - phi(len(self.rewards), (self.alpha / (2 * k)))
            self.UCB = self.mean + phi(len(self.rewards), (self.alpha / 2))

        elif method == 'bootstrap':
            if online:
                # If self.mean is still NaN, it is the first time we are constructing the confidence bounds
                if np.isnan(self.mean):

                    # Set all the counts to the length of the rewards
                    self.online_counts = [len(self.rewards_to_process) for i in range(iterations)]

                    # Take iterations number of bootstraps
                    bootstraps = list(
                        bs._generate_distributions([np.array(self.rewards_to_process)], iterations)[0])

                    # Calculate the mean for every bootstrap
                    self.online_means = [np.mean(bootstrap) for bootstrap in bootstraps]

                # If self.mean is not NaN, it means we already have online means.
                else:
                    bootstraps = deque(bs._generate_distributions([np.array(self.rewards_to_process)], len(self.online_means))[0])

                    for index, bootstrap in enumerate(bootstraps):
                        # Update the online mean
                        total_sum = (self.online_means[index] * self.online_counts[index]) + np.sum(bootstrap)
                        self.online_counts[index] += len(bootstrap)
                        self.online_means[index] = total_sum / self.online_counts[index]

                # Get the confidence interval
                confidence_interval = bs._get_confidence_interval(self.online_means,
                                                                  stat_val=bs_stats.mean(self.online_means)[0],
                                                                  alpha=self.alpha, is_pivotal=True)

                self.rewards.extend(self.rewards_to_process)
                self.rewards_to_process = []

            else: #not online
                self.rewards.extend(self.rewards_to_process)
                self.rewards_to_process = []
                confidence_interval = bs.bootstrap(np.array(self.rewards), stat_func=bs_stats.mean, alpha=self.alpha,
                                                       num_iterations=iterations,
                                                   iteration_batch_size=100, num_threads=1)

            self.mean = confidence_interval.value
            self.LCB = confidence_interval.lower_bound
            self.UCB = confidence_interval.upper_bound

    def sample(self):
        return np.random.beta(self.a, self.b, 1)[0]




