. ~/.bashrc
cd $CLIP_ROOT || exit
conda activate language_vision_bias || exit

cd references/scaling-laws-openclip/ || exit
python download_models.py
cd ../.. || exit

for i in {1..10}; do
  python measure_bias_and_performance/ndkl_maxskew.py & sleep 20
done

for i in {1..60}; do
  python measure_bias_and_performance/ndkl_maxskew.py & sleep 600
done
