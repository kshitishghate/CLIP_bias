. ~/.bashrc
cd $CLIP_ROOT || exit
conda activate language_vision_bias || exit

bash slurm_and_shell_scripts/clear_files.sh &

for i in {1..5}; do
  python measure_bias_and_performance/seat_replication.py & sleep 20
done

for i in {1..60}; do
  python measure_bias_and_performance/seat_replication.py & sleep 600
done
