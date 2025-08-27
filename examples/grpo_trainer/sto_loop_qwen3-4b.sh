#!/bin/bash

# Script to sequentially run data preprocessing and training for different stochastic probabilities and types
# Loop through p values: 0, 0.2, 0.4, 0.6, 0.8, 1.0
# Loop through stochastic types: wrong_to_right

set -e  # Exit on any error

# Base directory
HOME=/remote/alan3/verl
DATA_DIR="$HOME/data/gsm8k"

# Create log directory for this experiment
LOG_DIR="$HOME/logs/stochastic_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Starting stochastic probability and type experiment"
echo "Log directory: $LOG_DIR"

# Loop through stochastic types
for stochastic_type in "wrong_to_right"; do
    echo "=========================================="
    echo "Processing stochastic type: $stochastic_type"
    echo "=========================================="
    
    # Create subdirectory for this stochastic type
    TYPE_LOG_DIR="$LOG_DIR/type_${stochastic_type}"
    mkdir -p "$TYPE_LOG_DIR"
    
    # Loop through probability values
    for p in 0.0 0.2 0.4 0.6 0.8 1.0; do
    # for p in 1.0; do
        echo "=========================================="
        echo "Processing probability p = $p"
        echo "=========================================="
        
        # Create subdirectory for this probability
        PROB_LOG_DIR="$TYPE_LOG_DIR/prob_${p}"
        mkdir -p "$PROB_LOG_DIR"
        
        # Step 1: Run data preprocessing
        echo "Step 1: Running data preprocessing with stochastic_prob=$p, stochastic_type=$stochastic_type"
        python examples/data_preprocess/gsm8k.py \
            --local_dir "$DATA_DIR" \
            --stochastic_prob "$p" \
            --stochastic_type "$stochastic_type" \
            > "$PROB_LOG_DIR/preprocess.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "Data preprocessing completed successfully for p=$p, type=$stochastic_type"
        else
            echo "ERROR: Data preprocessing failed for p=$p, type=$stochastic_type"
            exit 1
        fi
        
        # Step 2: Start training job in background
        echo "Step 2: Starting training job for p=$p, type=$stochastic_type"
        
        # Create a temporary script for this specific run
        TEMP_SCRIPT="$PROB_LOG_DIR/train_script.sh"
        cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
cd $HOME
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=5fb2c3eb35cb3bc0124a02069ce91eedc6570e5a
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

python3 -m verl.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    data.train_files=$DATA_DIR/train.parquet \\
    data.val_files=$DATA_DIR/test.parquet \\
    data.train_batch_size=1024 \\
    data.max_prompt_length=512 \\
    data.max_response_length=1024 \\
    data.filter_overlong_prompts=True \\
    data.truncation='error' \\
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.actor.use_kl_loss=True \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
    actor_rollout_ref.actor.entropy_coeff=0 \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.rollout.n=1 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    algorithm.use_kl_in_reward=False \\
    trainer.critic_warmup=0 \\
    trainer.logger=['console','wandb'] \\
    trainer.project_name='verl_grpo_example_gsm8k' \\
    trainer.experiment_name="qwen3_4b_stochastic_p${p}_${stochastic_type}_clip_ratio_1000" \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes=1 \\
    trainer.save_freq=-1 \\
    trainer.test_freq=1 \\
    trainer.total_epochs=15
EOF
        
        chmod +x "$TEMP_SCRIPT"
        
        # Start the training job in background and capture PID
        nohup bash "$TEMP_SCRIPT" > "$PROB_LOG_DIR/training.log" 2>&1 &
        TRAIN_PID=$!
        
        echo "Training job started with PID: $TRAIN_PID"
        echo "Training logs will be saved to: $PROB_LOG_DIR/training.log"
        
        # Step 3: Wait for the training job to complete
        echo "Step 3: Waiting for training job to complete..."
        wait $TRAIN_PID
        
        if [ $? -eq 0 ]; then
            echo "Training completed successfully for p=$p, type=$stochastic_type"
        else
            echo "ERROR: Training failed for p=$p, type=$stochastic_type"
            echo "Check logs at: $PROB_LOG_DIR/training.log"
            exit 1
        fi
        
        echo "Completed processing for p=$p, type=$stochastic_type"
        echo ""
    done
    
    echo "Completed processing for stochastic type: $stochastic_type"
    echo ""
done

echo "=========================================="
echo "All experiments completed successfully!"
echo "All logs saved to: $LOG_DIR"
echo "=========================================="