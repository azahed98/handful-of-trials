from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np
from scipy.io import savemat

from .Controller import Controller
from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.optimizers import RandomOptimizer, CEMOptimizer


class MPC(Controller):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, params):
        """Creates class instance.

        Arguments:
            params
                .env (gym.env): Environment for which this controller will be used.
                .update_fns (list<func>): A list of functions that will be invoked
                    (possibly with a tensorflow session) every time this controller is reset.
                .ac_ub (np.ndarray): (optional) An array of action upper bounds.
                    Defaults to environment action upper bounds.
                .ac_lb (np.ndarray): (optional) An array of action lower bounds.
                    Defaults to environment action lower bounds.
                .per (int): (optional) Determines how often the action sequence will be optimized.
                    Defaults to 1 (reoptimizes at every call to act()).
                .prop_cfg
                    .model_init_cfg (DotMap): A DotMap of initialization parameters for the model.
                        .model_constructor (func): A function which constructs an instance of this
                            model, given model_init_cfg.
                    .model_train_cfg (dict): (optional) A DotMap of training parameters that will be passed
                        into the model every time is is trained. Defaults to an empty dict.
                    .model_pretrained (bool): (optional) If True, assumes that the model
                        has been trained upon construction.
                    .mode (str): Propagation method. Choose between [E, DS, TSinf, TS1, MM].
                        See https://arxiv.org/abs/1805.12114 for details.
                    .npart (int): Number of particles used for DS, TSinf, TS1, and MM propagation methods.
                    .ign_var (bool): (optional) Determines whether or not variance output of the model
                        will be ignored. Defaults to False unless deterministic propagation is being used.
                    .obs_preproc (func): (optional) A function which modifies observations (in a 2D matrix)
                        before they are passed into the model. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc (func): (optional) A function which returns vectors calculated from
                        the previous observations and model predictions, which will then be passed into
                        the provided cost function on observations. Defaults to lambda obs, model_out: model_out.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .obs_postproc2 (func): (optional) A function which takes the vectors returned by
                        obs_postproc and (possibly) modifies it into the predicted observations for the
                        next time step. Defaults to lambda obs: obs.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .targ_proc (func): (optional) A function which takes current observations and next
                        observations and returns the array of targets (so that the model learns the mapping
                        obs -> targ_proc(obs, next_obs)). Defaults to lambda obs, next_obs: next_obs.
                        Note: Only needs to process NumPy arrays.
                .opt_cfg
                    .mode (str): Internal optimizer that will be used. Choose between [CEM, Random].
                    .cfg (DotMap): A map of optimizer initializer parameters.
                    .plan_hor (int): The planning horizon that will be used in optimization.
                    .obs_cost_fn (func): A function which computes the cost of every observation
                        in a 2D matrix.
                        Note: Must be able to process both NumPy and Tensorflow arrays.
                    .ac_cost_fn (func): A function which computes the cost of every action
                        in a 2D matrix.
                .log_cfg
                    .save_all_models (bool): (optional) If True, saves models at every iteration.
                        Defaults to False (only most recent model is saved).
                        Warning: Can be very memory-intensive.
                    .log_traj_preds (bool): (optional) If True, saves the mean and variance of predicted
                        particle trajectories. Defaults to False.
                    .log_particles (bool) (optional) If True, saves all predicted particles trajectories.
                        Defaults to False. Note: Takes precedence over log_traj_preds.
                        Warning: Can be very memory-intensive
        """
        super().__init__(params)
        self.dO, self.dU = params.env.observation_space.shape[0], params.env.action_space.shape[0]
        self.ac_ub, self.ac_lb = params.env.action_space.high, params.env.action_space.low
        self.ac_ub = np.minimum(self.ac_ub, params.get("ac_ub", self.ac_ub))
        self.ac_lb = np.maximum(self.ac_lb, params.get("ac_lb", self.ac_lb))
        self.update_fns = params.get("update_fns", [])
        self.per = params.get("per", 1)

        self.model = get_required_argument(
            params.prop_cfg.model_init_cfg, "model_constructor", "Must provide a model constructor."
        )(params.prop_cfg.model_init_cfg)
        self.model_train_cfg = params.prop_cfg.get("model_train_cfg", {})
        self.prop_mode = get_required_argument(params.prop_cfg, "mode", "Must provide propagation method.")
        self.npart = get_required_argument(params.prop_cfg, "npart", "Must provide number of particles.")
        self.ign_var = params.prop_cfg.get("ign_var", False) or self.prop_mode == "E"

        self.obs_preproc = params.prop_cfg.get("obs_preproc", lambda obs: obs)
        self.obs_postproc = params.prop_cfg.get("obs_postproc", lambda obs, model_out: model_out)
        self.obs_postproc2 = params.prop_cfg.get("obs_postproc2", lambda next_obs: next_obs)
        self.targ_proc = params.prop_cfg.get("targ_proc", lambda obs, next_obs: next_obs)

        self.opt_mode = get_required_argument(params.opt_cfg, "mode", "Must provide optimization method.")
        self.plan_hor = get_required_argument(params.opt_cfg, "plan_hor", "Must provide planning horizon.")
        self.adap_hor = params.opt_cfg.get("adap_hor", None)
        self.plan_min = params.opt_cfg.get("plan_min", 10)
        self.iter_plan_hor = self.plan_min
        self.adap_param = params.opt_cfg.get("adap_param", 15.0)
        self.prev_uncert = -1
        self.cur_tot_uncert = 0
        self.cur_count = 0

        self.obs_cost_fn = get_required_argument(params.opt_cfg, "obs_cost_fn", "Must provide cost on observations.")
        self.ac_cost_fn = get_required_argument(params.opt_cfg, "ac_cost_fn", "Must provide cost on actions.")

        self.save_all_models = params.log_cfg.get("save_all_models", False)
        self.log_traj_preds = params.log_cfg.get("log_traj_preds", False)
        self.log_particles = params.log_cfg.get("log_particles", False)

        # Perform argument checks
        if self.prop_mode not in ["E", "DS", "MM", "TS1", "TSinf"]:
            raise ValueError("Invalid propagation method.")
        if self.prop_mode in ["TS1", "TSinf"] and self.npart % self.model.num_nets != 0:
            raise ValueError("Number of particles must be a multiple of the ensemble size.")
        if self.prop_mode == "E" and self.npart != 1:
            raise ValueError("Deterministic propagation methods only need one particle.")

        # Create action sequence optimizer
        opt_cfg = params.opt_cfg.get("cfg", {})
        self.optimizer = MPC.optimizers[params.opt_cfg.mode](
            sol_dim=self.plan_hor*self.dU,
            lower_bound=np.tile(self.ac_lb, [self.plan_hor]),
            upper_bound=np.tile(self.ac_ub, [self.plan_hor]),
            tf_session=None if not self.model.is_tf_model else self.model.sess,
            **opt_cfg
        )

        # Controller state variables
        self.has_been_trained = params.prop_cfg.get("model_pretrained", False)
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.plan_hor_buf = 0
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])
        self.train_in = np.array([]).reshape(0, self.dU + self.obs_preproc(np.zeros([1, self.dO])).shape[-1])
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dO]), np.zeros([1, self.dO])).shape[-1]
        )
        if self.model.is_tf_model:
            self.sy_cur_obs = tf.Variable(np.zeros(self.dO), dtype=tf.float32)
            self.ac_seq = tf.placeholder(shape=[1, self.plan_hor*self.dU], dtype=tf.float32)
            self.pred_cost, self.pred_traj, _, __ = self._compile_cost(self.ac_seq, get_pred_trajs=True)
            self.optimizer.setup(self._compile_cost, True)
            self.model.sess.run(tf.variables_initializer([self.sy_cur_obs]))
        else:
            raise NotImplementedError()

        print("Created an MPC controller, prop mode %s, %d particles. " % (self.prop_mode, self.npart) +
              ("Ignoring variance." if self.ign_var else ""))

        if self.save_all_models:
            print("Controller will save all models. (Note: This may be memory-intensive.")
        if self.log_particles:
            print("Controller is logging particle predictions (Note: This may be memory-intensive).")
            self.pred_particles = []
        elif self.log_traj_preds:
            print("Controller is logging trajectory prediction statistics (mean+var).")
            self.pred_means, self.pred_vars = [], []
        else:
            print("Trajectory prediction logging is disabled.")

    def train(self, obs_trajs, acs_trajs, rews_trajs):
        """Trains the internal model of this controller. Once trained,
        this controller switches from applying random actions to using MPC.

        Arguments:
            obs_trajs: A list of observation matrices, observations in rows.
            acs_trajs: A list of action matrices, actions in rows.
            rews_trajs: A list of reward arrays.

        Returns: None.
        """
        # Construct new training points and add to training set
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            new_train_in.append(np.concatenate([self.obs_preproc(obs[:-1]), acs], axis=-1))
            new_train_targs.append(self.targ_proc(obs[:-1], obs[1:]))
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)

        # Train the model
        self.model.train(self.train_in, self.train_targs, **self.model_train_cfg)
        self.has_been_trained = True

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()
        if self.model.is_tf_model:
            for update_fn in self.update_fns:
                update_fn(self.model.sess)

    def act(self, obs, t, get_pred_cost=False):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        if not self.has_been_trained:
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape), 0
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            plan_hor = self.plan_hor_buf
            return action, plan_hor

        if self.model.is_tf_model:
            self.sy_cur_obs.load(obs, self.model.sess)

        soln, plan_hor, uncert = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        self.cur_tot_uncert += uncert
        self.cur_count += 1
        self.prev_sol = np.concatenate([np.copy(soln)[self.per*self.dU:], np.zeros(self.per*self.dU)])
        self.ac_buf = soln[:min(self.per, plan_hor)*self.dU].reshape(-1, self.dU)
        print(self.ac_buf.shape)
        self.plan_hor_buf = plan_hor

        if get_pred_cost and not (self.log_traj_preds or self.log_particles):
            if self.model.is_tf_model:
                pred_cost = self.model.sess.run(
                    self.pred_cost,
                    feed_dict={self.ac_seq: soln[None]}
                )[0]
            else:
                raise NotImplementedError()
            action, plan_hor = self.act(obs, t)
            return action, plan_hor, pred_cost
        elif self.log_traj_preds or self.log_particles:
            pred_cost, pred_traj = self.model.sess.run(
                [self.pred_cost, self.pred_traj],
                feed_dict={self.ac_seq: soln[None]}
            )
            pred_cost, pred_traj = pred_cost[0], pred_traj[:, 0]
            if self.log_particles:
                self.pred_particles.append(pred_traj)
            else:
                self.pred_means.append(np.mean(pred_traj, axis=1))
                self.pred_vars.append(np.mean(np.square(pred_traj - self.pred_means[-1]), axis=1))
            if get_pred_cost:
                action, plan_hor = self.act(obs, t)
                return action, plan_hor, pred_cost
        action, plan_hor = self.act(obs, t)
        return action, plan_hor

    def dump_logs(self, primary_logdir, iter_logdir):
        """Saves logs to either a primary log directory or another iteration-specific directory.
        See __init__ documentation to see what is being logged.

        Arguments:
            primary_logdir (str): A directory path. This controller assumes that this directory
                does not change every iteration.
            iter_logdir (str): A directory path. This controller assumes that this directory
                changes every time dump_logs is called.

        Returns: None
        """
        self.model.save(iter_logdir if self.save_all_models else primary_logdir)
        if self.log_particles:
            savemat(os.path.join(iter_logdir, "predictions.mat"), {"predictions": self.pred_particles})
            self.pred_particles = []
        elif self.log_traj_preds:
            savemat(
                os.path.join(iter_logdir, "predictions.mat"),
                {"means": self.pred_means, "vars": self.pred_vars}
            )
            self.pred_means, self.pred_vars = [], []

    def _compile_cost(self, ac_seqs, get_pred_trajs=False):
        t, nopt = tf.constant(0), tf.shape(ac_seqs)[0]
        init_costs = tf.zeros([nopt, self.npart])
        ac_seqs = tf.reshape(ac_seqs, [-1, self.plan_hor, self.dU])
        ac_seqs = tf.reshape(tf.tile(
            tf.transpose(ac_seqs, [1, 0, 2])[:, :, None],
            [1, 1, self.npart, 1]
        ), [self.plan_hor, -1, self.dU])
        init_obs = tf.tile(self.sy_cur_obs[None], [nopt * self.npart, 1])
        popsize = int(int(init_obs.shape[0])/self.npart)

        def continue_prediction(t, *args):
            return tf.less(t, self.plan_hor)
        def continue_prediction_iterative(t, *args):
            return tf.less(t, self.iter_plan_hor)
        def continue_prediction_heuristic(t, total_cost, cur_obs, pred_trajs, uncertainties, *args):
            uncertainty_avg = tf.reduce_mean(uncertainties)
            h = tf.cast(1.0/(tf.log(.5 * uncertainty_avg + 1)), tf.int32)
            h = tf.maximum(h, self.plan_min)
            return tf.math.reduce_all([tf.less(t, h), tf.less(t, self.plan_hor)])

        pred_trajs = init_obs[None]

        def iteration(t, total_cost, cur_obs, pred_trajs, uncertainties=None, all_costs=None):
            cur_acs = ac_seqs[t]
            if self.adap_hor == "heuristic" or self.adap_hor == "iterative":
                next_obs, total, aleatoric, epistemic = self._predict_next_obs(cur_obs, cur_acs, return_uncertainties=True)
                uncertainties_shape = uncertainties.shape

                uncertainties = tf.cond(tf.less(t, 1), 
                                        lambda: tf.reshape(total, [-1, int((popsize * self.npart) /self.model.num_nets)]), 
                                        lambda: uncertainties)
                uncertainties = tf.reshape(uncertainties, [-1, uncertainties_shape[1]])

                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + delta_cost, next_obs, pred_trajs, uncertainties

            elif self.adap_hor == "adaptive":
                next_obs, total, aleatoric, epistemic = self._predict_next_obs(cur_obs, cur_acs, return_uncertainties=True)
                uncertainties = tf.concat([uncertainties, tf.minimum(total[None], self.adap_param)], axis=0)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                total_cost = total_cost + delta_cost
                all_costs = tf.concat([all_costs, total_cost[None]], axis=0)
                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t+1, total_cost, next_obs, pred_trajs, uncertainties, all_costs
            elif self.adap_hor is None:
                next_obs = self._predict_next_obs(cur_obs, cur_acs)
                delta_cost = tf.reshape(
                    self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs), [-1, self.npart]
                )
                next_obs = self.obs_postproc2(next_obs)
                pred_trajs = tf.concat([pred_trajs, next_obs[None]], axis=0)
                return t + 1, total_cost + delta_cost, next_obs, pred_trajs
            
            else:
                raise NotImplementedError


        loop_vars = [t, init_costs, init_obs, pred_trajs]
        shape_invariants = [t.get_shape(), init_costs.get_shape(), init_obs.get_shape(), tf.TensorShape([None, None, self.dO])]

        cont_fn = continue_prediction
        if self.adap_hor:
            uncertainties = tf.zeros(shape=[0,init_obs.shape[0]])
            loop_vars.append(uncertainties)
            shape_invariants.append(tf.TensorShape([None, uncertainties.shape[1]]))
            cont_fn = continue_prediction_heuristic
        if self.adap_hor == "adaptive":
            all_costs = tf.zeros(shape=[0, init_costs.shape[0], init_costs.shape[1]])
            loop_vars.append(all_costs)
            shape_invariants.append(tf.TensorShape([None, all_costs.shape[1], all_costs.shape[2]]))
            cont_fn = continue_prediction
        if self.adap_hor == "iterative":
            cont_fn = continue_prediction_iterative

        while_ret = tf.while_loop(
            cond=cont_fn, body=iteration, loop_vars=loop_vars,
            shape_invariants=shape_invariants
        )
        if self.adap_hor == "heuristic" or self.adap_hor == "iterative":
            t, costs, cur_obs, pred_trajs, uncertainties = while_ret
            t = tf.Print(t, [t], message="Trajectory length")
            # costs = costs/ tf.cast(t, tf.float32)
        elif self.adap_hor == "adaptive":
            t, costs, cur_obs, pred_trajs, uncertainties, all_costs = while_ret
        elif self.adap_hor is None:
            t, costs, cur_obs, pred_trajs = while_ret
        else:
            raise NotImplementedError
        
        if self.adap_hor == "adaptive":
            cum_uncert = tf.math.cumsum(uncertainties, axis=0)[self.plan_min:, :]

            cropped_trajs = tf.reshape(pred_trajs[self.plan_min + 1:], [-1, self.dO])
            # uncert_mask = tf.cast(tf.greater(cum_uncert, 
            #                                  tf.maximum(tf.contrib.distributions.percentile(cum_uncert[0], 
            #                                                                       .9,  
            #                                                                       interpolation="higher",
            #                                                                       keep_dims=True), self.adap_param)
            #                                  ) ,tf.bool)
            uncert_mask = tf.cast(tf.greater(cum_uncert, 
                                             tf.maximum(tf.reduce_max(cum_uncert[0]),  self.adap_param)
                                            ), tf.bool)
            # uncert_mask = tf.cast(tf.greater(cum_uncert, tf.maximum(tf.reduce_max(cum_uncert[0]), .55)),tf.bool)
            # uncert_mask = tf.concat([tf.zeros_like(uncert_mask[0])[None,:], uncert_mask[1:, :]], axis=0)
            
            uncert_mask = tf.Print(uncert_mask, [tf.reduce_sum(tf.cast(uncert_mask, tf.float32), axis=0),
                                                 cum_uncert[:,15],
                                                 tf.reduce_mean(tf.reduce_sum(tf.cast(uncert_mask, tf.float32), axis=0))], 
                                    message="Uncertainty mask")

            all_costs = tf.reshape(all_costs, [-1, nopt * self.npart])
            divisor = tf.tile(tf.range(1, self.plan_hor+1, dtype=tf.float32)[self.plan_min:, None], [1, nopt * self.npart])
            all_costs = all_costs[self.plan_min:, :]
            all_costs = tf.where(uncert_mask, -1e6 * tf.ones_like(all_costs), all_costs)

            horizons = tf.reduce_max((1-tf.cast(uncert_mask, dtype=tf.int32))*tf.cast(divisor, dtype=tf.int32), axis=0)
            avg_horizon = tf.reduce_mean(horizons)
            all_costs = tf.Print(all_costs, 
                                 [avg_horizon, tf.reduce_max(horizons)],
                                 message="Average, max horizon")

            all_costs = tf.tile(tf.reduce_max(all_costs, axis=0)[None, :], [self.plan_hor - self.plan_min, 1])

            all_costs = tf.where(uncert_mask, 1e6 * tf.ones_like(all_costs), all_costs)

            costs = tf.reduce_min(all_costs, axis=0)[None, :] / tf.cast(horizons, tf.float32)
            costs = tf.reshape(costs, [nopt, self.npart])
        
        # Replace nan costs with very high cost
        costs = tf.reduce_mean(tf.where(tf.is_nan(costs), 1e6 * tf.ones_like(costs), costs), axis=1)
        if get_pred_trajs:
            pred_trajs = tf.reshape(pred_trajs, [self.plan_hor + 1, -1, self.npart, self.dO])
            if self.adap_hor == "adaptive":
                return costs, pred_trajs, avg_horizon, tf.reduce_mean(uncertainties)
            elif self.adap_hor == "iterative":
                return costs, pred_trajs, self.iter_plan_hor, tf.reduce_mean(uncertainties)
            return costs, pred_trajs, self.plan_hor, 0
        else:
            if self.adap_hor == "adaptive":
                return costs, avg_horizon, tf.reduce_mean(uncertainties)
            elif self.adap_hor == "iterative":
                return costs, self.iter_plan_hor, tf.reduce_mean(uncertainties)
            return costs, self.plan_hor, 0

    def _predict_next_obs(self, obs, acs, return_uncertainties=False):
        proc_obs = self.obs_preproc(obs)

        if self.model.is_tf_model:
            # TS Optimization: Expand so that particles are only passed through one of the networks.
            if self.prop_mode == "TS1":
                proc_obs = tf.reshape(proc_obs, [-1, self.npart, proc_obs.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    tf.random_uniform([tf.shape(proc_obs)[0], self.npart]),
                    k=self.npart
                ).indices
                tmp = tf.tile(tf.range(tf.shape(proc_obs)[0])[:, None], [1, self.npart])[:, :, None]
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                proc_obs = tf.gather_nd(proc_obs, idxs)
                proc_obs = tf.reshape(proc_obs, [-1, proc_obs.get_shape()[-1]])
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                proc_obs, acs = self._expand_to_ts_format(proc_obs), self._expand_to_ts_format(acs)

            # Obtain model predictions
            inputs = tf.concat([proc_obs, acs], axis=-1)
            mean, var = self.model.create_prediction_tensors(inputs)

            if return_uncertainties:
                # uncert_inputs = tf.reshape(inputs, [self.model.num_nets, inputs.shape[1]/self.npart, self.npart, ])
                uncert_inputs = tf.reshape(inputs, [-1, inputs.get_shape()[-1]])
                uncert_mean, uncert_var = self.model.create_prediction_tensors(uncert_inputs, factored=True)

                total, aleatoric, epistemic = VarianceUncertainty(uncert_mean, uncert_var, return_separated=True)
                
                total = tf.reduce_mean(tf.reshape(total, [self.npart, -1]), axis=0)
                total = tf.reshape(tf.tile(total[None, :], [self.npart, 1]), [-1])

                aleatoric = tf.reduce_mean(tf.reshape(aleatoric, [self.npart, -1]), axis=0)
                aleatoric = tf.reshape(tf.tile(aleatoric[None, :], [self.npart, 1]), [-1])

                epistemic = tf.reduce_mean(tf.reshape(epistemic, [self.npart, -1]), axis=0)
                epistemic = tf.reshape(tf.tile(epistemic[None, :], [self.npart, 1]), [-1])

            if self.model.is_probabilistic and not self.ign_var:
                predictions = mean + tf.random_normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var)
                if self.prop_mode == "MM":
                    model_out_dim = predictions.get_shape()[-1].value

                    predictions = tf.reshape(predictions, [-1, self.npart, model_out_dim])
                    prediction_mean = tf.reduce_mean(predictions, axis=1, keep_dims=True)
                    prediction_var = tf.reduce_mean(tf.square(predictions - prediction_mean), axis=1, keep_dims=True)
                    z = tf.random_normal(shape=tf.shape(predictions), mean=0, stddev=1)
                    samples = prediction_mean + z * tf.sqrt(prediction_var)
                    predictions = tf.reshape(samples, [-1, model_out_dim])
            else:
                predictions = mean

            # TS Optimization: Remove additional dimension
            if self.prop_mode == "TS1" or self.prop_mode == "TSinf":
                predictions = self._flatten_to_matrix(predictions)
            if self.prop_mode == "TS1":
                predictions = tf.reshape(predictions, [-1, self.npart, predictions.get_shape()[-1]])
                sort_idxs = tf.nn.top_k(
                    -sort_idxs,
                    k=self.npart
                ).indices
                idxs = tf.concat([tmp, sort_idxs[:, :, None]], axis=-1)
                predictions = tf.gather_nd(predictions, idxs)
                predictions = tf.reshape(predictions, [-1, predictions.get_shape()[-1]])

            if return_uncertainties:
                return self.obs_postproc(obs, predictions), total, aleatoric, epistemic

            return self.obs_postproc(obs, predictions)
        else:
            raise NotImplementedError()

    def _expand_to_ts_format(self, mat):
        dim = mat.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(mat, [-1, self.model.num_nets, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [self.model.num_nets, -1, dim]
        )

    def _flatten_to_matrix(self, ts_fmt_arr):
        dim = ts_fmt_arr.get_shape()[-1]
        return tf.reshape(
            tf.transpose(
                tf.reshape(ts_fmt_arr, [self.model.num_nets, -1, self.npart // self.model.num_nets, dim]),
                [1, 0, 2, 3]
            ),
            [-1, dim]
        )

def VarianceEpistemicUncertainty(means):
    """
    Arguments:
        means: TF Tensor of shape [ensemble_size, batch_size, dimension]. Means of Gaussians
        vars: TF Tensor of shape [ensemble_size, batch_size, dimension]. Variances of Gaussians
    Returns:
        TF Tensor of shape [btach_size] for the epistemic uncertainy as measured by Variance
        In our case, we use Var(X) = E[ || X - E[ X ] ||_2^2 ], which is expected squared euclidean distance from the mean
    """
    variances = tf.nn.moments(means, axes=[0])[1]
    collapsed_variance = tf.reduce_sum(variances, axis=1)
    return collapsed_variance

def VarianceAleatoricUncertainty(vars):
    """
    Arguments:
        means: TF Tensor of shape [ensemble_size, batch_size, dimension]. Means of Gaussians
        vars: TF Tensor of shape [ensemble_size, batch_size, dimension]. Variances of Gaussians
    Returns:
        TF Tensor of shape [btach_size] for the aleatoric uncertainy as measured by Variance
        In our case, we use Var(X) = E[ || X - E[ X ] ||_2^2 ], which is expected squared euclidean distance from the mean
    """
    collapsed_variance = tf.reduce_sum(vars, axis=2)
    reduced_mean = tf.reduce_mean(collapsed_variance, axis=0)
    return reduced_mean

def VarianceUncertainty(means, vars, return_separated=False):
    """
    Arguments:
        means: TF Tensor of shape [ensemble_size, batch_size, dimension]. Means of Gaussians
        vars: TF Tensor of shape [ensemble_size, batch_size, dimension]. Variances of Gaussians
        return_separated: If true eturns total, epistemic and aleatoric uncertainty
    Returns:
        TF Tensors of shape [btach_size] for the total uncertainy as measured by Variance
        In our case, we use Var(X) = E[ || X - E[ X ] ||_2^2 ], which is expected squared euclidean distance from the mean
    """
    aleatoric = VarianceAleatoricUncertainty(vars) 
    epistemic = VarianceEpistemicUncertainty(means)
    total = aleatoric + epistemic
    if return_separated:
        return total, aleatoric, epistemic
    return total
