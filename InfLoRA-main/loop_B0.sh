#!/bin/bash
for i in $(seq 1 2); do
  case $i in
    1) prefix="1st" ;;
    2) prefix="2nd" ;;
    *) prefix="${i}th" ;;
  esac

  # 后台执行
  nohup ./train.sh > "./res/${prefix}-inflora-B0-prototype-and-first.out" 2>&1 &

  # 等待本次后台进程执行完再继续下一次循环
  wait
done
