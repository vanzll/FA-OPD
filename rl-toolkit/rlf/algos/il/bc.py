import copy

import gym
import numpy as np
import time
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn.functional as F
from rlf.algos.il.base_il import BaseILAlgo
from rlf.args import str2bool
from rlf.storage.base_storage import BaseStorage
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np


class BehavioralCloning(BaseILAlgo):
    """
    When used as a standalone updater, BC will perform a single update per call
    to update. The total number of udpates is # epochs * # batches in expert
    dataset. num-steps must be 0 and num-procs 1 as no experience should be collected in the
    environment. To see the performance, you must evaluate. To just evaluate at
    the end of training, set eval-interval to a large number that is greater
    than the number of updates. There will always be a final evaluation.
    """

    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, args):
        super().init(policy, args)
        self.num_epochs = 0
        self.action_dim = rutils.get_ac_dim(self.policy.action_space)
        if self.args.bc_state_norm:
            self.norm_mean = self.expert_stats["state"][0]
            self.norm_var = torch.pow(self.expert_stats["state"][1], 2)
        else:
            self.norm_mean = None
            self.norm_var = None
        self.num_bc_updates = 0

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            print("Setting environment state normalization")
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs_x = torch.clamp(
            (rutils.get_def_obs(x) - self.norm_mean)
            / torch.pow(self.norm_var + 1e-8, 0.5),
            -10.0,
            10.0,
        )
        if isinstance(x, dict):
            x["observation"] = obs_x
            return x
        else:
            return obs_x

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def first_train(self, log, eval_policy, env_interface):
        """Create initial log records before BC training starts."""
        print("BC training started; creating initial log records...")
        
        # Record initial metrics at the start of training.
        initial_metrics = {
            'step': 0,
            'episode': 0,
            'epoch': 0,
            'training_progress': 0.0,
            'action_loss': 0.0,
            'timestamp': time.time(),
            'wall_time': time.time(),
            # Initial placeholders before evaluation is available.
            'avg_r': 0.0,  
            'avg_ep_found_goal': 0.0,
            'dist_entropy': 0.0,
        }
        
        # Write initial metrics to the logger.
        log.log_vals(initial_metrics, 0)
        print("BC initial log records created")
        
        # Call the parent first_train hook if available.
        if hasattr(super(), 'first_train'):
            super().first_train(log, eval_policy, env_interface)
        
        # Keep evaluation references for post-training evaluation.
        self._log = log
        self._eval_policy = eval_policy
        self._env_interface = env_interface

    def full_train(self, update_iter=0):
        action_loss = []
        prev_num = 0

        # BC training loop.
        print(f"Starting BC training ({self.args.bc_num_epochs} epochs)")
        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step(False)
                action_loss.append(log_vals["_pr_action_loss"])

                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        # Save the training loss curve.
        rutils.plot_line(
            action_loss,
            f"action_loss_{update_iter}.png",
            self.args.vid_dir,
            not self.args.no_wb,
            self.get_completed_update_steps(self.update_i),
        )
        
        # Evaluate performance after training.
        print("BC training completed; starting performance evaluation...")
        self._post_training_evaluation()
        # Circle visualization for FP baseline
        self._plot_circle_fp()
        
        self.num_epochs = 0

    def _post_training_evaluation(self):
        """Evaluate performance after training and record the result."""
        if not hasattr(self, '_eval_policy') or not hasattr(self, '_log'):
            print("Evaluation function is unavailable; skipping performance evaluation")
            return
            
        try:
            # Create evaluation arguments.
            import copy
            eval_args = copy.copy(self.args)
            eval_args.eval_num_processes = min(20, getattr(self.args, 'eval_num_processes', 10))
            eval_args.num_eval = getattr(self.args, 'num_eval', 100)
            eval_args.num_render = 0
            
            print(f"Evaluating policy performance with {eval_args.num_eval} episodes...")
            
            # Run evaluation.
            tmp_env = self._eval_policy(self.policy, self.num_bc_updates, True, eval_args)
            
            # Close the temporary environment.
            if tmp_env is not None:
                tmp_env.close()
            
            print("BC performance evaluation completed")
            
        except Exception as e:
            print(f"BC performance evaluation failed: {e}")
            # Record a final completion metric even if evaluation fails.
            final_metrics = {
                'step': self.num_bc_updates,
                'episode': 0,
                'epoch': self.num_epochs,
                'training_progress': 1.0,
                'timestamp': time.time(),
                'wall_time': time.time(),
                'bc_training_complete': 1
            }
            
            try:
                self._log.log_vals(final_metrics, self.num_bc_updates)
                print("BC training completion status recorded")
            except:
                print("Unable to record final BC training status")

    def _plot_circle_fp(self):
        """For Circle-v0: scatter predicted actions along s in [-1,1] and overlay unit circle."""
        env_name = str(getattr(self.args, 'env_name', ''))
        if not env_name.startswith('Circle'):
            return
        try:
            device = getattr(self.args, 'device', 'cpu')
            s = np.linspace(-1.0, 1.0, 400, endpoint=True).astype(np.float32)
            states = torch.from_numpy(s).view(-1, 1).to(device)
            self.policy.eval()
            with torch.no_grad():
                actions, _, _ = self.policy(states, None, None)
                a = actions.squeeze(-1).detach().cpu().numpy()

            os.makedirs('./data/imgs', exist_ok=True)
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.scatter(s, a, c='#8e44ad', s=10, alpha=0.8, label='FP predictions')
            theta = np.linspace(0.0, 2.0 * np.pi, 400)
            ax.plot(np.cos(theta), np.sin(theta), c='black', lw=1.2, label='s^2 + a^2 = 1')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.05])
            ax.set_xlabel('state s')
            ax.set_ylabel('action a')
            ax.legend()
            ax.grid(True, ls='--', alpha=0.3)
            plt.tight_layout()
            out_fp = os.path.join('./data/imgs', f"{self.args.prefix}_fp_circle.png")
            plt.savefig(out_fp)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] FP circle plot failed: {e}")

    def pre_update(self, cur_update):
        # Override the learning rate decay
        pass

    def _bc_step(self, decay_lr):
        # import ipdb; ipdb.set_trace
        if decay_lr:
            super().pre_update(self.num_bc_updates)
        expert_batch = self._get_next_data()

        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()

        states, true_actions = self._get_data(expert_batch)

        log_dict = {}

        pred_actions, _, _ = self.policy(states, None, None)
        if rutils.is_discrete(self.policy.action_space):
            pred_label = rutils.get_ac_compact(self.policy.action_space, pred_actions)
            acc = (pred_label == true_actions.long()).sum().float() / pred_label.shape[
                0
            ]
            log_dict["_pr_acc"] = acc.item()
            log_dict["acc"] = acc.item()  # standard accuracy metric
        
        loss = autils.compute_ac_loss(
            pred_actions,
            true_actions.view(-1, self.action_dim),
            self.policy.action_space,
        )

        self._standard_step(loss)
        self.num_bc_updates += 1

        val_loss = self._compute_val_loss()
        if val_loss is not None:
            log_dict["_pr_val_loss"] = val_loss.item()
            log_dict["val_loss"] = val_loss.item()  # standard validation loss metric

        log_dict["_pr_action_loss"] = loss.item()
        log_dict["action_loss"] = loss.item()  # standard action loss metric
        
        # Add standard training-progress metrics.
        log_dict["step"] = self.num_bc_updates
        log_dict["episode"] = 0  # BC has no episode concept.
        log_dict["epoch"] = self.num_epochs
        log_dict["training_progress"] = self.num_epochs / max(1, self.args.bc_num_epochs)
        log_dict["timestamp"] = time.time()

        return log_dict

    def _get_data(self, batch):
        states = batch["state"].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = batch["actions"].to(self.args.device)
        true_actions = self._adjust_action(true_actions)
        return states, true_actions

    def _compute_val_loss(self):
        if self.update_i % self.args.eval_interval != 0:
            return None
        if self.val_train_loader is None:
            return None
        with torch.no_grad():
            losses = []
            for batch in self.val_train_loader:
                states, true_actions = self._get_data(batch)
                pred_actions, _, _ = self.policy(states, None, None)
                loss = autils.compute_ac_loss(
                    pred_actions,
                    true_actions.view(-1, self.action_dim),
                    self.policy.action_space,
                )
                losses.append(loss.item())

            return np.mean(losses)

    def update(self, storage, args, beginning, t):
        top_log_vals = super().update(storage)
        log_vals = self._bc_step(True)
        log_vals.update(top_log_vals)
        return log_vals

    def get_storage_buffer(self, policy, envs, args):
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            # This is set when BC is used at the same time as another optimizer
            # that also has a learning rate.
            self.set_arg_prefix("bc")

        super().get_add_args(parser)
        #########################################
        # Overrides
        if self.set_arg_defs:
            parser.add_argument("--num-processes", type=int, default=1)
            parser.add_argument("--num-steps", type=int, default=0)
            ADJUSTED_INTERVAL = 200
            parser.add_argument("--log-interval", type=int, default=ADJUSTED_INTERVAL)
            parser.add_argument(
                "--save-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
            parser.add_argument(
                "--eval-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
        parser.add_argument("--no-wb", default=False, action="store_true")

        #########################################
        # New args
        parser.add_argument("--bc-num-epochs", type=int, default=1)
        parser.add_argument("--bc-state-norm", type=str2bool, default=False)
        parser.add_argument("--bc-noise", type=float, default=None)
