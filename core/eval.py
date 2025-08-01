import jax.numpy as jnp
import jax
import numpy as np
import tqdm
import ml_collections
import sys
from absl import flags

from models.qwen3 import create_model_from_ckpt
from utils.configs import define_flag_dict
from envs.env_creator import create_env
from envs.countdown import extract_xml_strategy
from utils.sharding import create_sharding, host_gather
from models.tokenizer import create_tokenizer
from core.sampling import pad_and_collate, autoregressive_sample


def eval_model(model, params, env, 
               num_generation_tokens,
               force_answer_at,
               prompt_length,
               inference_batch_per_device,
               pad_id,
               shard_data_fn,
               no_shard,
               data_shard,
               num_epochs
               ):
    np.random.seed(jax.process_index())
    host_id = jax.process_index()
    env_num_tasks = env.num_tasks if env.num_tasks != -1 else 100
    total_num_tasks = num_epochs * env_num_tasks
    env_task_idx = 0
    rollout_batch_size = jax.local_device_count() * inference_batch_per_device
    global_batch_size = rollout_batch_size * jax.process_count()
    rng = jax.random.PRNGKey(jax.process_index())

    env_infos_history = {}
    env_infos_history['return'] = []
    for i in tqdm.tqdm(range(total_num_tasks // global_batch_size + 1)):
        env_states, env_tokens = [], []
        for _ in range(rollout_batch_size):
            env_state, output_tokens = env.reset(min(env_task_idx + jax.process_index(), env_num_tasks-1))
            env_task_idx += jax.process_count()
            env_task_idx = env_task_idx % env_num_tasks
            env_states.append(env_state)
            env_tokens.append(output_tokens)

        prompt_tokens = pad_and_collate(env_tokens, pad_id=pad_id, force_length=prompt_length)
        prompt_tokens = shard_data_fn(prompt_tokens)
        num_generation_tokens = num_generation_tokens
        rng, key = jax.random.split(rng)
        action_tokens = autoregressive_sample(
            model, params, prompt_tokens, rng=key, num_generation_tokens=num_generation_tokens,
            pad_id=pad_id, data_shard=data_shard, no_shard=no_shard, force_answer_at=force_answer_at,
        )
        prompt_tokens = host_gather(prompt_tokens)
        action_tokens = host_gather(action_tokens)
        action_tokens_local = action_tokens[host_id * rollout_batch_size : (host_id+1) * rollout_batch_size]
        new_states, _, returns_local, dones, env_infos = env.step_list(env_states, [t.tolist() for t in action_tokens_local])
        assert dones[0] # Only supports bandit envs for now.
        returns_local = np.array(returns_local)
        returns = host_gather(shard_data_fn(returns_local))
        for k, v in env_infos.items():
            if k not in env_infos_history:
                env_infos_history[k] = []
            v_global = host_gather(shard_data_fn(np.array(v)))
            env_infos_history[k] += v_global.tolist()
        env_infos_history['return'] += returns.tolist()
    env_infos_history = {k: np.array(v)[:total_num_tasks] for k, v in env_infos_history.items()}
    return new_states, env_infos_history



def eval_proposer_model(proposer, proposer_params,
                        model, params,
                        env,
                        tokenizer,
                        num_generation_tokens,
                        force_answer_at,
                        proposer_prompt_length,
                        prompt_length,
                        inference_batch_per_device,
                        pad_id,
                        shard_data_fn,
                        no_shard,
                        data_shard,
                        model_shard_data_fn,
                        model_no_shard,
                        model_data_shard,
                        num_epochs
                        ):
    np.random.seed(jax.process_index())
    host_id = jax.process_index()
    env_num_tasks = env.num_tasks if env.num_tasks != -1 else 100
    total_num_tasks = num_epochs * env_num_tasks
    env_task_idx = 0
    rollout_batch_size = jax.local_device_count() * inference_batch_per_device
    global_batch_size = rollout_batch_size * jax.process_count()
    rng = jax.random.PRNGKey(jax.process_index())

    env_infos_history = {}
    env_infos_history['return'] = []
    for i in tqdm.tqdm(range(total_num_tasks // global_batch_size + 1)):
        env_states, env_tokens, env_proposer_tokens = [], [], []
        for _ in range(rollout_batch_size):
            env_state, output_tokens, proposer_tokens = env.reset(min(env_task_idx + jax.process_index(), env_num_tasks-1))
            env_task_idx += jax.process_count()
            env_task_idx = env_task_idx % env_num_tasks
            env_states.append(env_state)
            env_tokens.append(output_tokens)
            env_proposer_tokens.append(proposer_tokens)

        proposer_prompt_tokens = pad_and_collate(env_proposer_tokens, pad_id=pad_id,
                                                 force_length=proposer_prompt_length)
        proposer_prompt_tokens = shard_data_fn(proposer_prompt_tokens)
        # prompt_tokens = shard_data_fn(prompt_tokens)
        num_generation_tokens = num_generation_tokens
        rng, proposer_key, model_key = jax.random.split(rng, 3)
        proposer_action_tokens, _ = autoregressive_sample(
            proposer, proposer_params, proposer_prompt_tokens,
            rng=proposer_key, num_generation_tokens=num_generation_tokens,
            pad_id=pad_id, data_shard=data_shard, no_shard=no_shard,
        )
        proposer_action_tokens = host_gather(proposer_action_tokens)
        for idx, tokens in enumerate(proposer_action_tokens):
            cleaned_proposer_action_tokens = env.clean_action(tokens.tolist(), tokenizer.get_eos_token_id())
            proposer_action_msg = tokenizer.decode(cleaned_proposer_action_tokens)
            strategy = extract_xml_strategy(proposer_action_msg)

            env_msg = tokenizer.decode(env_tokens[idx])
            env_msg = env_msg.replace("<strategy> None </strategy>",
                                      "<strategy> {strategy} </strategy>".format(strategy=strategy))
            env_tokens[idx] = tokenizer.encode(env_msg)
        prompt_tokens = pad_and_collate(env_tokens, pad_id=pad_id, force_length=prompt_length)
        prompt_tokens = model_shard_data_fn(prompt_tokens)
        action_tokens = autoregressive_sample(
            model, params, prompt_tokens,
            rng=model_key, num_generation_tokens=num_generation_tokens,
            pad_id=pad_id, data_shard=model_data_shard, no_shard=model_no_shard,
            force_answer_at=force_answer_at,
        )
        prompt_tokens = host_gather(prompt_tokens)
        action_tokens = host_gather(action_tokens)
        action_tokens_local = action_tokens[host_id * rollout_batch_size : (host_id+1) * rollout_batch_size]
        new_states, _, returns_local, dones, env_infos = env.step_list(env_states, [t.tolist() for t in action_tokens_local])
        assert dones[0] # Only supports bandit envs for now.
        returns_local = np.array(returns_local)
        returns = host_gather(shard_data_fn(returns_local))
        for k, v in env_infos.items():
            if k not in env_infos_history:
                env_infos_history[k] = []
            v_global = host_gather(shard_data_fn(np.array(v)))
            env_infos_history[k] += v_global.tolist()
        env_infos_history['return'] += returns.tolist()
    env_infos_history = {k: np.array(v)[:total_num_tasks] for k, v in env_infos_history.items()}
    return new_states, env_infos_history



######$###########################################
### Runnable function to eval a checkpoint on an env.
##################################################
if __name__ == '__main__':
    config = ml_collections.ConfigDict({
        'model_dir': '/nfs/gcs/jaxconverted/Qwen3-1.7B/',
        # env settings.
        'env_name': 'poem',
        'num_generation_tokens': -1, # -1 = use default from env.
        'force_answer_at': -1, # -1 = use default from env.
        'prompt_length': 256, # Length of the prompt to pad to.
        'num_epochs': 1,
        # sampling settings.
        'inference_batch_per_device': 4, # Set this to the maximum until OOM. Should not affect results.
    })
    define_flag_dict(config)
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    ckpt_dir = FLAGS.model_dir
    model, params = create_model_from_ckpt(ckpt_dir)
    rng = jax.random.PRNGKey(0)
    params_shard, no_shard, data_shard, shard_data_fn = create_sharding('fsdp', params)
    params = jax.jit(lambda x: x, out_shardings=params_shard)(params)

    jax.debug.visualize_array_sharding(params['Block_0']['Dense_0']['kernel'])
    tokenizer = create_tokenizer(ckpt_dir)
    pad_id = tokenizer.get_pad_token_id()

    env = create_env(FLAGS.env_name, tokenizer)
    if FLAGS.num_generation_tokens == -1:
        FLAGS.num_generation_tokens = env.tokens_per_action
    if FLAGS.force_answer_at == -1:
        FLAGS.force_answer_at = env.force_answer_at

    new_states, env_infos_history = eval_model(
        model, params, env,
        num_generation_tokens=FLAGS.num_generation_tokens,
        force_answer_at=FLAGS.force_answer_at,
        prompt_length=FLAGS.prompt_length,
        inference_batch_per_device=FLAGS.inference_batch_per_device,
        pad_id=pad_id,
        shard_data_fn=shard_data_fn,
        no_shard=no_shard,
        data_shard=data_shard,
        num_epochs=FLAGS.num_epochs,
    )
    print(" ======================= Example Rollout ======================= ")
    print(env.render(new_states[0]))
    print(" =============================================================== ")
    print("Number of rollouts:", len(env_infos_history['return']))
    for k, v in env_infos_history.items():
        print(f"{k}: {np.mean(v)}")