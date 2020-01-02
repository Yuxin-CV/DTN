# Diversity Transfer Network

Pytorch implementation for "Diversity Transfer Network for Few-Shot Learning" (deep backbone, on miniImageNet).
We also provide our trainded model.

## Dataset Preprocessing & Evaluate our trained model on miniImageNet

Download the dataset from [this link](https://drive.google.com/open?id=1XapMobTsCSw9gyySt9D0GF_hOX_XpeZx), put the `images` folder in `./miniImageNet/`.

Then run:

```bash
bash make.sh
```

## Train your DTN on miniImageNet

```python
python main_DTN.py --checkpoint 'your_checkpoint'
```

## Evaluate your DTN on miniImageNet
```python
# 5-way 5-shot
python main_DTN.py --N-way 5 --N-shot 5 --evaluate 1 --resume 'your_checkpoint/checkpoint.pth.tar'
# 5-way 1-shot
python main_DTN.py --N-way 5 --N-shot 1 --evaluate 1 --resume 'your_checkpoint/checkpoint.pth.tar'
```
## Acknowledgment
[Horizon Robotics](http://en.horizon.ai/)

## License

## Citations
If you find DTN useful in your research, please consider citing:
```
@inproceedings{Chen2019DiversityTN,
  title={Diversity Transfer Network for Few-Shot Learning},
  author={Mengting Chen and Yuxin Fang and Xinggang Wang and Heng Luo and Yifeng Geng and Xinyu Zhang and Chang Huang and Wenyu Liu and Bo Wang},
  year={2019}
}
```
## Thanks to the Third Party Libs
[Pytorch](https://github.com/pytorch/pytorch)   