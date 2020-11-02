#!/usr/bin/env bash

set -e
set -x


declare -A source_datasets=(
    ["personachat"]="personachat"
    ["opensub"]="OpenSubtitles2018"
    ["daily_dialog"]="dailydialog"
)

declare -A personachat_models=(
    ["seq2seq"]="host-172-20-181-14_gpu0/seq2seq/personachat_h3/original/model.opt"
    ["cvae"]="host-172-20-181-14_gpu0/cvae/personachat_h3/original/model.opt"
    ["transformer"]="host-172-20-181-14_gpu1/transformer/personachat_h3/original/model.opt"
    ["hred"]="host-172-20-181-14_gpu3/hred/personachat_h3/original/model.opt"
    ["dialogwae"]="host-172-20-181-14_gpu2/dialogwae/personachat_h3/original/model.opt"
)

declare -A daily_dialog_models=(
    ["seq2seq"]="host-172-20-181-14_gpu2/seq2seq/daily_dialog/original/model.opt"
    ["cvae"]="host-172-20-181-12_gpu3/cvae/daily_dialog/original/model.opt"
    ["transformer"]="host-172-20-181-14_gpu2/transformer/daily_dialog/original/model.opt"
    ["hred"]="host-172-20-181-12_gpu1/hred/daily_dialog/original/model.opt"
    ["dialogwae"]="host-172-20-181-16_gpu3/dialogwae/daily_dialog/original/model.opt"
)

declare -A opensub_models=(
    ["seq2seq"]="host-172-20-181-14_gpu1/seq2seq/opensub_h3_sparse_small/original/model.opt"
    ["cvae"]="host-172-20-181-14_gpu3/cvae/opensub_h3_sparse_small/original/model.opt"
    ["transformer"]="host-172-20-181-14_gpu3/transformer/opensub_h3_sparse_small/original/model.opt"
    ["hred"]="host-172-20-181-12_gpu2/hred/opensub_h3_sparse_small/original/model.opt"
    ["dialogwae"]="host-172-20-181-12_gpu0/dialogwae/opensub_h3_sparse_small/original/model.opt"
)

# -------------------------- Mandatory Arguments ---------------------
history_size=3
sparse=False
task_type="personachat"

# make sure that the data_path is compatible with the configuration above
data_path="AdaptiveLearning/personachat_history3"

# relative path of multi_turn dialog data to build from
source_multi_turn_data_dir=${PARLAI_HOME}/data/AdaptiveLearning/source_multi_turn_data/${source_datasets[$task_type]}
# -------------------------- Mandatory Arguments ---------------------

# make sure that this array is always untouched
declare -a source_multi_turn_data_types=("train.txt" "valid.txt" "test.txt")

declare -a data_sizes=(-1 -1 -1)
declare -a score_funcs=(
    "intrep_word" "avg_nidf" "lastuttsim" "post_sim"
    "loss_of_seq2seq" "loss_of_cvae" "loss_of_transformer" "loss_of_hred" "loss_of_dialogwae"
)

print_next_post=True  # default to be True
build_arora=True
build_nidf=True
correct_train_file=True  # remove lines of train.txt that contains in valid.txt or test.txt

out_dir=${PARLAI_HOME}/data/${data_path}
if [[ ! -d "$out_dir" ]]; then
    mkdir -p ${out_dir}
fi

cd ${PARLAI_HOME}
n=${#source_multi_turn_data_types[@]}
for ((i=0; i<$n;i++)); do
    data_type=${source_multi_turn_data_types[$i]}

    python projects/adaptive_learning/scripts/build_single_turn_fb_dialog.py \
        --fb_dialog_file ${source_multi_turn_data_dir}/${data_type}  \
        --history_size ${history_size} \
        --sparse ${sparse} \
        --print_next_post ${print_next_post} \
        > /tmp/$(basename ${source_multi_turn_data_dir})_${data_type}_his${history_size}.txt

    if [[ ${data_sizes[$i]} -gt 0 ]]; then
        shuf -o /tmp/$(basename ${source_multi_turn_data_dir})_${data_type}_his${history_size}.txt \
        -n ${data_sizes[$i]} \
        < /tmp/$(basename ${source_multi_turn_data_dir})_${data_type}_his${history_size}.txt
    fi

    # save to ${out_dir}
    cat /tmp/$(basename ${source_multi_turn_data_dir})_${data_type}_his${history_size}.txt \
    | python projects/adaptive_learning/scripts/format_fb_dialog.py > ${out_dir}/${data_type}.tmp

    cat ${out_dir}/${data_type}.tmp | awk -F$'\t' '!seen[$1, $2]++' > ${out_dir}/${data_type}
    /bin/rm -rf ${out_dir}/${data_type}.tmp
done

if [[ "${build_arora}" = True ]] ; then
    # build the arora and nidf data
    python projects/adaptive_learning/scripts/arora.py \
        --arora_data_dir ${data_path} \
        --include_context ${sparse}
fi

if [[ "${build_nidf}" = True ]] ; then
    python projects/adaptive_learning/scripts/nidf.py \
        --nidf_data_dir ${data_path} \
        --include_context ${sparse}
fi

function correct_file()
{
    local file_under_corrected=$1

    # correct the file
    if [[ ${correct_train_file} = True ]]; then
        cd ${PARLAI_HOME}/data/${data_path}
        python ${PARLAI_HOME}/projects/adaptive_learning/scripts/correct_train_data.py \
            --train_file ${file_under_corrected} \
            --valid_file valid.txt \
            --test_file test.txt > ${file_under_corrected}.corrected
        mv -f ${file_under_corrected}.corrected ${file_under_corrected}
        cd -
    fi
}

correct_file ${PARLAI_HOME}/data/${data_path}/train.txt

# build the task data for training
for func_name in "${score_funcs[@]}"
do
    train_file=${PARLAI_HOME}/data/${data_path}/train.txt
    print_next_post=False

    if [[ ${func_name} == "post_sim" ]]
    then
        train_file=/tmp/$(basename ${source_multi_turn_data_dir})_${source_multi_turn_data_types[0]}_his${history_size}.txt

        # uniq it regarding the first two columns
        cat ${train_file} | awk -F$'\t' '!seen[$1, $2]++' > /tmp/train_file_for_post_sim.txt
        train_file=/tmp/train_file_for_post_sim.txt

        # correct the file
        correct_file ${train_file}
    fi

    args="--arora_data_dir ${data_path} \
          --nidf_data_dir ${data_path} \
          --task_dir ${data_path} \
          --train_file ${train_file} \
          --print_next_post ${print_next_post} "

    if [[ "${func_name}" == "loss_of_"* ]]; then
        model_name=$(echo ${func_name}| cut -d'_' -f 3)
        model_opt=${task_type}_models[${model_name}]
        args=${args}"  --score_func loss \
        --model_name ${model_name} --model_opt_path ${PARLAI_HOME}/models/adaptive_learning_v1/${!model_opt} "
    else
        args=${args}"  --score_func ${func_name}"
    fi

    python projects/adaptive_learning/scripts/make_data.py ${args}

    cd ${PARLAI_HOME}/data/${data_path}

    suffix="${func_name}"

    if [[ ! -d "${suffix}" ]]; then
        mkdir ${suffix}
    fi

    mv train.txt.${suffix} ${suffix}
    cat ${suffix}/train.txt.${suffix} |awk -F$'\t' '!seen[$1, $2]++' |sort -t $'\t' -k3,3 -g > ${suffix}/train.txt.${suffix}.sorted
    ln -svf train.txt.${suffix}.sorted ${suffix}/train.txt
    ln -svf ../valid.txt ${suffix}/valid.txt
    ln -svf ../test.txt ${suffix}/test.txt
    cd -
    # now, we are in the directory ${PARLAI_HOME}
done

# build the task data for analyzing
if [[ ! -d ${PARLAI_HOME}/data/${data_path}/data4analyse ]]
then
    mkdir ${PARLAI_HOME}/data/${data_path}/data4analyse
fi

cd ${PARLAI_HOME}/data/${data_path}
for func_name in "${score_funcs[@]}"
do
    suffix=${func_name}
    if [[ ${suffix} == "lastuttsim" ]] || [[ ${suffix} == "post_sim" ]]
    then
        echo "For lastuttsim or post_sim, use reverse_norm."
        cat ${suffix}/train.txt.${suffix} | awk -F$'\t' '{print $3}' | \
        python ${PARLAI_HOME}/projects/adaptive_learning/scripts/norm_score.py \
        --reverse True > data4analyse/${suffix}.norm
    elif [[ "${suffix}" == "loss_of_"* ]]
    then
        echo "For model loss, we normalize it (min-max)"
        cat ${suffix}/train.txt.${suffix} | awk -F$'\t' '{print $3}' | \
        python ${PARLAI_HOME}/projects/adaptive_learning/scripts/norm_score.py \
        > data4analyse/${suffix}.norm
    else
        echo "For avg_nidf or intrep_word, just copy the 3rd column to stdout."
        cat ${suffix}/train.txt.${suffix} | awk -F$'\t' '{print $3}' > \
        data4analyse/${suffix}.norm
    fi
done
cd -

# build the original task
cd ${PARLAI_HOME}/data/${data_path}

if [[ ! -d "original" ]]; then
    mkdir original
fi

ln -svf ../train.txt original/train.txt
ln -svf ../valid.txt original/valid.txt
ln -svf ../test.txt original/test.txt
cd -

cd ${PARLAI_HOME}