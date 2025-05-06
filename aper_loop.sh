#!/bin/bash
for i in $(seq 1 3); do
  case $i in
    1) prefix="1st" ;;
    2) prefix="2nd" ;;
    3) prefix="3rd" ;;
    *) prefix="${i}th" ;;
  esac

  # 后台执行
  nohup ./scripts/train_aper.sh > "./res/${prefix}-aper-B0-lora16.out" 2>&1 &

  # 等待本次后台进程执行完再继续下一次循环
  wait
done
