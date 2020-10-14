#!/bin/bash

optimizer=adam
learning_rate=1e-3
batch_size=1
eval_batch_size=1
dim=512
num_reasoning_hops=64
lr_scheduler=reduceonplateau
warmup_updates=1000
output_dir="/scratch/jassiene/MacNetTextExperiments"
datapath=/scratch/jassiene/data

model_name=m_${optimizer}_lr_${learning_rate}_bs_${batch_size}_dim_${dim}_nrh_${num_reasoning_hops}_lrscheduler_${lr_scheduler}_wu_${warmup_updates}

export PYTHONPATH=$PYTHONPATH:/home/jassiene
agent_path=~/parlai_internal/agents/mac_net/
rm -fr $agent_path
mkdir -p $agent_path
touch ~/parlai_internal/__init__.py
touch ~/parlai_internal/agents/__init__.py
cp -R ./*.py $agent_path

python -m parlai.scripts.multiprocessing_train --datapath $datapath --tensorboard_log=True  --tensorboard-log True --train-predict True -stim 120 -m internal:mac_net -t qangaroo \
     -bs $batch_size --eval_batchsize $eval_batch_size -veps 3 -mf "${output_dir}/$model_name" -nrh $num_reasoning_hops -dim $dim --optimizer $optimizer --lr_scheduler $lr_scheduler \
     --learningrate $learning_rate --warmup_updates $warmup_updates

# python3 ~/ParlAI/parlai/scripts/train_model.py --tensorboard_log=True  --tensorboard-log True --train-predict True -stim 120 -m mac_net -t qangaroo \
#      -bs $batch_size -veps 3 -mf "${output_dir}/$model_name" -nrh $num_reasoning_hops -dim $dim --optimizer $optimizer --lr_scheduler $lr_scheduler \
#      --learningrate $learning_rate --warmup_updates $warmup_updates