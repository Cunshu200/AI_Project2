# bash 02_train_vae.sh

for i in {1..150}
do
    python 02_train_vae.py --split_i $i
    sleep 1.0
done