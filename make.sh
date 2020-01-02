echo "Dataset preprocessing..."
cd ./miniImageNet
python proc_images.py
python img2pickle.py
cd ..

echo "Evaluate our trained model..."
echo "5-way 1-shot"
CUDA_VISIBLE_DEVICES=1 python main_DTN.py --N-way 5 --N-shot 1 --evaluate 1 --resume 'DTN_SEED#3/checkpoint.pth.tar'
echo "5-way 5-shot"
CUDA_VISIBLE_DEVICES=1 python main_DTN.py --N-way 5 --N-shot 5 --evaluate 1 --resume 'DTN_SEED#3/checkpoint.pth.tar'