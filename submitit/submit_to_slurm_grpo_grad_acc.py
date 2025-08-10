import os
from pathlib import Path
from datetime import datetime
import platform

import submitit


def main():
    cluster_name = platform.node().split('-')[0]
    if cluster_name == 'adroit':
        log_root_dir = '/home/cz8792/network'
        partition = 'gpu'
        account = None
        nodelist = None
        constraint = None
    elif 'della' in cluster_name:
        log_root_dir = '/home/cz8792/gpfs'
        partition = 'gpu-short'
        account = None
        nodelist = None
        constraint = "gpu80"
        # log_root_dir = '/home/cz8792/gpfs'
        # partition = 'pli'
        # account = 'rlchongyiz'
        # nodelist = None
        # constraint = None
    elif cluster_name in ['soak.cs.princeton.edu', 'wash.cs.princeton.edu',
                          'rinse.cs.princeton.edu', 'spin.cs.princeton.edu']:
        log_root_dir = '/n/fs/rl-chongyiz'
        partition = None
        account = 'pnlp'
        nodelist = None
        constraint = None
    elif cluster_name == 'neuronic.cs.princeton.edu':
        log_root_dir = '/n/fs/prl-chongyiz'
        partition = 'all'
        account = None
        nodelist = None
        constraint = None
    else:
        raise NotImplementedError

    executor = submitit.AutoExecutor(folder="/tmp/submitit_logs")  # this path is not actually used.
    executor.update_parameters(
        slurm_name="grpo_grad_acc",
        slurm_time=int(24 * 60),  # minute
        slurm_partition=partition,
        slurm_account=account,
        slurm_nodes=1,
        slurm_ntasks_per_node=1,  # tasks can share nodes
        slurm_cpus_per_task=8,
        slurm_mem="200G",
        slurm_gpus_per_node=4,
        slurm_nodelist=nodelist,
        slurm_stderr_to_stdout=True,
        slurm_array_parallelism=20,
        slurm_constraint=constraint,
    )

    with executor.batch():  # job array
        for env_name in ['countdown']:
            for num_steps in [160]:
                for inference_batch_per_device in [18]:
                    for ppo_minibatch in [64]:  # 64
                        for grad_accum_steps in [8]:
                            for entropy_coef in [0, 0.001]:
                                for seed in [10, 20]:
                                    exp_name = f"{datetime.today().strftime('%Y%m%d')}_grpo_env_name={env_name}_num_steps={num_steps}_inference_batch_per_device={inference_batch_per_device}_ppo_minibatch={ppo_minibatch}_grad_accum_steps={grad_accum_steps}_entropy_coef={entropy_coef}"
                                    log_dir = os.path.expanduser(
                                        f"{log_root_dir}/exp_logs/jax_llm_logs/grpo_grad_acc/{exp_name}/{seed}")

                                    # change the log folder of slurm executor
                                    submitit_log_dir = os.path.join(os.path.dirname(log_dir), 'submitit')
                                    executor._executor.folder = Path(
                                        submitit_log_dir).expanduser().absolute()

                                    cmds = f"""
                                        unset PYTHONPATH;
                                        source $HOME/.zshrc;
                                        conda activate jax_llm;
                                        which python;
                                        echo $CONDA_PREFIX;
                
                                        echo job_id: $SLURM_ARRAY_JOB_ID;
                                        echo task_id: $SLURM_ARRAY_TASK_ID;
                                        squeue -j $SLURM_JOB_ID -o "%.18i %.9P %.8j %.8u %.2t %.6D %.5C %.11m %.11l %.12N";
                
                                        export PROJECT_DIR=$PWD;
                                        export PYTHONPATH=$HOME/research/lmpo;
                                        export PATH="$PATH":"$CONDA_PREFIX"/bin;
                                        export CUDA_VISIBLE_DEVICES=0,1,2,3;
                                        export WANDB_CACHE_DIR={log_root_dir}/.cache/wandb;
                                        source $HOME/env_vars.sh;
                                        XLA_PYTHON_CLIENT_MEM_FRACTION=.95;
                                        
                                        rm -rf {log_dir};
                                        mkdir -p {log_dir};
                                        python $PROJECT_DIR/core/grpo_grad_acc.py \
                                            --env_name={env_name} \
                                            --test_env_name={env_name} \
                                            --model_dir=/scratch/gpfs/cz8792/language_models/models/Qwen--Qwen3-1.7B/jax_ckpts/ \
                                            --inference_batch_per_device={inference_batch_per_device} \
                                            --num_steps={num_steps} \
                                            --ppo_minibatch={ppo_minibatch} \
                                            --grad_accum_steps={grad_accum_steps} \
                                            --entropy_coef={entropy_coef} \
                                            --save_dir={log_dir} \
                                            --seed={seed} \
                                        2>&1 | tee {log_dir}/stream.log;
                
                                        export SUBMITIT_RECORD_FILENAME={log_dir}/submitit_"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID".txt;
                                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_submitted.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_submission.sh" >> "$SUBMITIT_RECORD_FILENAME";
                                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_log.out" >> "$SUBMITIT_RECORD_FILENAME";
                                        echo "{submitit_log_dir}/"$SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID"_0_result.pkl" >> "$SUBMITIT_RECORD_FILENAME";
                                    """

                                    cmd_func = submitit.helpers.CommandFunction([
                                        "/bin/zsh", "-c",
                                        cmds,
                                    ], verbose=True)

                                    executor.submit(cmd_func)


if __name__ == "__main__":
    main()
