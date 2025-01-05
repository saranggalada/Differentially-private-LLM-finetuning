# Differentially-private-LLM-finetuning

Simply clone and run the `build.sh` file

It automatically clones / pulls the rest of the repo, installs python if not already, creates an environment on the GPU, installs libraries from `requirements.txt` and executes the code `dp-llm-finetune.py` with configurations from `dp-finetune-config.json`

Once the code completes execution, the finetuned model is pushed automatically to the huggingface repo set in the config file (https://huggingface.co/Sarang-Galada/dp-llm-finetune)
