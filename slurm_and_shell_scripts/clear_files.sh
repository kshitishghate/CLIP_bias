  . ~/.bashrc
cd $CLIP_ROOT || exit
conda activate language_vision_bias || exit

for i in {1..4000}; do
  find references/scaling-laws-openclip/ -name epoch* -mmin +12 -delete
  find references/scaling-laws-openclip/ -name tmp* -mmin +12 -delete
  sleep 20
done
