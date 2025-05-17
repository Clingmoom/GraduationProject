#!/bin/bash

# ✅ 强制要求第一个参数存在（任务名）
if [ -z "$1" ]; then
    echo "❌ 错误：必须传入任务名（如 [sft|ppo|test]"
    exit 1
fi

# 📅 构造分支名（按当天日期）
DATE_STR=$(date +%F)
BRANCH_NAME="dengcaifang/$DATE_STR"
DATE_CN=$(date -d "$DATE_STR" +'%Y年%m月%d日')
echo "📅 当前日期：$DATE_CN"
echo "🌿 切换到远程分支：$BRANCH_NAME"

# 🔧 Git 操作
git fetch origin
git reset --hard origin/$BRANCH_NAME || {
    echo "❌ 分支不存在：origin/$BRANCH_NAME"
    exit 1
}
TASK_NAME=$1

echo "启动本任务的TensorBoard"
ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
mkdir -p ./logs
tensorboard --port 6007 --logdir "./logs" --bind_all

shift  # 去掉第一个参数，剩下的全是给 Python 的参数

echo "🚀 正在运行任务：$TASK_NAME _train"

if [ "$TASK_NAME" == "sft" ]; then
    python ./src/train_stage_1.py "$@"
elif [ "$TASK_NAME" == "ppo" ]; then
    python ./src/train_stage_2.py "$@"
else
    echo "❌ 错误：未知任务名 $TASK_NAME，只支持 [sft|ppo|test]"
    exit 1
fi


#source run.sh [sft|ppo|test]
#torchrun --standalone --nproc_per_node=1 ./src/train_stage_${TASK_NAME}.py