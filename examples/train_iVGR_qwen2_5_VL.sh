# LLM judge endpoint (used by the reward function for accuracy / consistency scoring)
export JUDGE_BASE_URL=http://10.20.32.193:12345/v1
export JUDGE_API_KEY=EMPTY
export JUDGE_MODEL_NAME=judge

# Optional: separate endpoint for consistency scoring (defaults to the judge endpoint)
# export CONSISTENCY_BASE_URL=http://10.20.32.192:12346/v1
# export CONSISTENCY_API_KEY=EMPTY

python3 -m verl.trainer.main config=examples/config_dual_group_training.yaml \
    trainer.experiment_name=ivgr_qwen2_5_vl_grpo \
    worker.actor.model.model_path=allencbzhang/iVGR-Qwen2.5-VL-7B-SFT  \
    trainer.save_freq=20 \
    trainer.save_limit=20 \
    trainer.max_steps=360 \
    trainer.total_epochs=-1 \
    worker.rollout.n=5 \
    data.max_response_length=4096 \
    data.paired_ratio=0.9 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.reward.reward_function_kwargs.json_log_dir=./checkpoints/ivgr_qwen2_5_vl_grpo/jsonlogs/