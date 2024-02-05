from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "/mnt/data/zhanghui/output-ckpts/lvlm/v5/finetune_lora_5/checkpoint-700",
    device_map="auto",
    trust_remote_code=True
).eval()
merged_model = model.merge_and_unload()
new_model_directory = "/mnt/data/zhanghui/output-ckpts/lvlm/v5/finetune_lora_5/chat_700_2"
merged_model.save_pretrained(new_model_directory, max_shard_size="2GB", safe_serialization=True)

