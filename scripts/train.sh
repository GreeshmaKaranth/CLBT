train()
{
main=CLBT/supervised_bert.py

vocab=CLBT/trial_data/multi_vocab.txt
vocab1=CLBT/trial_data/en_vocab.txt

batch=64
n_epoch=500
loss=l2_dist
opt='adam,lr=0.01'
max_sent=100
n_layer=2
ap='first'

gpu=$1
input0=$2
input1=$3
model=$4
align=$5
type=$6

if [ -z $1 ];then
  echo "usage:./train.sh [GPU] [bert file0] [bert file1] [model path] [align] [type]"
  exit
fi

#source $env 
CUDA_VISIBLE_DEVICES=$gpu python $main --bert_file0 $input0 --bert_file1 $input1 \
--model_path $model --batch_size $batch --n_epochs $n_epoch \
--map_optimizer $opt --n_layers $n_layer \
--load_pred_bert --n_max_sent $max_sent \
--align_file $align --loss $loss --map_type $type \
--align_punc --align_policy $ap \
--vocab_file $vocab --vocab_file1 $vocab1
}

env=/users2/yxwang/work/env/py3.6_torch0.4.1/bin/activate
# train 0 CLBT/trial_data/de-en.100.de.bert CLBT/trial_data/de-en.100.en.bert CLBT/models/test/svd.en-de.trial-model CLBT/trial_data/de-en.100.wp.align svd
train 0 CLBT/trial_data/en-hi.8.en.bert CLBT/trial_data/en-hi.8.hi.bert CLBT/models/test/svd.en-hi.trial-model CLBT/trial_data/en-hi.8.wp.align svd

