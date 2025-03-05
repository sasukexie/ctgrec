Read me

install LLaMA Factory

````
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
````

````
```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```
````

Datasets:  /data/resource

preprocess:  /tests/preprocess/chain

The /data and /tests data can be downloaded from here:https://drive.google.com/drive/folders/1bySWmDulEZ125QKe1e_ykR5HsGgnTe3-?usp=drive_link

Below you can find the configuration file:  /examples/rec

You can run with the following command:

```
pretrain:  nohup llamafactory-cli train examples/rec/llama3/lora_pretrain_ml.yaml > log/llama3/ml/pt_`date +"%Y%m%d%H%M%S"`.log 2>&1 &

sft:  nohup llamafactory-cli train examples/rec/llama3/lora_sft_ft_ml.yaml > log/llama3/ml/ft_`date +"%Y%m%d%H%M%S"`.log 2>&1 &

predict: nohup llamafactory-cli train examples/rec/llama3/lora_predict_ml.yaml > log/llama3/ml/pd_`date +"%Y%m%d%H%M%S"`.log 2>&1 &
```

