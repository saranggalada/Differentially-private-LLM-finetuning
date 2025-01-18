# Differentially-private-LLM-finetuning

Steps:

1. Edit `HF_READ_TOKEN` and `HF_WRITE_TOKEN` in `dp-finetune-config.json` with your huggingface tokens.
2. Clone the repo and run `build.sh` file using the command `bash build.sh`

The `build.sh` script automatically clones / pulls the rest of the repo, installs python if not already, creates an environment on the GPU, installs libraries from `requirements.txt` and executes the code `dp-llm-finetune.py` with configurations from `dp-finetune-config.json`

Once the code completes execution, the finetuned model is pushed automatically to the huggingface repo set in the config file
