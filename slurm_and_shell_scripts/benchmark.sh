. ~/.bashrc
conda activate language_vision_bias || exit
# Run 60 instances of same script in parallel
cd $CLIP_ROOT || exit
bash results/performance_evaluation/measure_performance.sh &
sleep 10
#bash results/performance_evaluation/measure_performance.sh &
#sleep 10
#bash results/performance_evaluation/measure_performance.sh &
#sleep 10
#bash results/performance_evaluation/measure_performance.sh &
#sleep 10
#bash results/performance_evaluation/measure_performance.sh &
wait


#
#for i in {1..60}; do
#  echo "Starting instance $i"
#  bash results/performance_evaluation/measure_performance.sh &
#  sleep 10
#done
