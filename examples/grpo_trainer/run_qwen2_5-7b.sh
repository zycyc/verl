set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
HOME=/remote/alan/verl
export WANDB_API_KEY=5fb2c3eb35cb3bc0124a02069ce91eedc6570e5a

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

uv run -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/examples/processed_data/train.parquet \
    data.val_files=$HOME/examples/processed_data/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='reflection' \
    trainer.experiment_name='qwen2.5_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2000 \
    use_memory=true \
    memory.embedding_model="all-MiniLM-L6-v2" \
    memory.similarity_threshold=0.95 \
    memory.save_freq=100 \
    memory.load_path=null $@