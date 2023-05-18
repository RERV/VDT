# for (( i=2; i<=600; i+=10 ))
# do
# echo $i
# python sample.py --image-size 128 --seed $i --data-path /home/haoyu_lu/video_dataset/physion/SlotFormer-master/data/Physion
# done

# python sample.py --image-size 128  --data-path /home/haoyu_lu/video_dataset/physion/SlotFormer-master/data/Physion
# torchrun --nproc_per_node=8 physion_fvd_ssim.py --image-size 128  --data-path /home/haoyu_lu/video_dataset/physion/SlotFormer-master/data/Physion
# for (( i=2; i<=15; i+=1 ))
# do
# echo $i
# torchrun --nproc_per_node=8 ddp_sample.py --condition_frames $i --image-size 128  --data-path /home/haoyu_lu/video_dataset/Cityscapes128_h5
# done

torchrun --nproc_per_node=8 ddp_sample.py --condition_frames 2 --image-size 128  --data-path /home/haoyu_lu/video_dataset/Cityscapes128_h5
