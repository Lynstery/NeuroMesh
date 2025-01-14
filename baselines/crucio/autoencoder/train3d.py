import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from crucio.autoencoder.dataset import (CNN_FRAME_NUM, VIDEO_DIR, KINETICS400_DIR, ReshapeVideo,
                                        VideoDataset, KineticsVideoDataset)
from crucio.autoencoder.loss import loss_function
from crucio.autoencoder.network3d import (get_networks3d,
                                          print_autoencoder3d_info)
from crucio.autoencoder.gpu import WORLD_SIZE, ddp_setup, print_device_info

# Define hyperparameter
use_kinetics_dataset = 1
sampler_rate = 0.1
num_epochs = 100 
save_epoch = 10
loss_index = 2
# Learning rate is positively correlated with batch size
if loss_index < 3:
    batch_size = 2
    learning_rate = WORLD_SIZE * 5e-4
else:
    batch_size = 1
    learning_rate = WORLD_SIZE * 5e-5


def main(rank, train_dataset):
    ddp_setup(rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, sampler=DistributedSampler(train_dataset), num_workers=8)
    if rank == 0:
        print(
            f'Length of training set (i.e. number of batch) is {len(train_loader)}')

    # Define model and optimizer
    encoder3d, encoder3d_path, decoder3d, decoder3d_path = get_networks3d(
        is_load=True, rank=rank, transfer="4.2.16.5")
    params = [{'params': net.parameters()} for net in [encoder3d, decoder3d]]
    optimizer = optim.Adam(params, lr=learning_rate)
    # Multiply learning rate by 0.1 every 12 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    # Define loss function
    criterion = loss_function(loss_index, rank)

    # Training model
    rev = ReshapeVideo()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        batch_index = 1
        train_loader.sampler.set_epoch(epoch)
        for i, (videos, inputs, paths) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward propagation
            videos = videos.to(rank)
            inputs = inputs.to(rank)
            codes = encoder3d(inputs)
            outputs = decoder3d(codes)

            # Calculate loss (and accuracy)
            output_frames = rev(outputs)
            video_frames = rev(videos)
            if loss_index < 3:
                loss = criterion(output_frames, video_frames)
            else:
                loss, acc = criterion(output_frames, video_frames)
                acc = torch.tensor(acc).to(rank)
                dist.all_reduce(acc)
                acc /= WORLD_SIZE
            dist.all_reduce(loss)
            loss /= WORLD_SIZE

            # Error propagation and optimization
            loss.backward()
            optimizer.step()
            del videos, inputs
            torch.cuda.empty_cache()
            train_loss += loss.item()
            if loss_index >= 3:
                train_acc += acc.item()
            if rank == 0:
                print(
                    f"Epoch [{epoch+1}] batch [{batch_index}/{len(train_loader)}]", end="\r")
            batch_index += 1

        # Automatically adjust learning rate
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        if rank == 0:
            if loss_index < 3:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Loss: {train_loss/len(train_loader):.4f}')
            else:
                print(
                    f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc/len(train_loader):.4f}')

        if (epoch+1) % save_epoch == 0:
            if rank == 0:
                print(f"Epoch [{epoch+1}], saving trained weights")
                torch.save(encoder3d.module.state_dict(), encoder3d_path)
                torch.save(decoder3d.module.state_dict(), decoder3d_path)

    if rank == 0:
        torch.save(encoder3d.module.state_dict(), encoder3d_path)
        torch.save(decoder3d.module.state_dict(), decoder3d_path)
    destroy_process_group()


if __name__ == "__main__":
    print_device_info()
    print_autoencoder3d_info()
    # Preprocess dataset
    # preprocess_video_dataset(VIDEO_DIR)
    # Prepare training dataset
    if use_kinetics_dataset:
        train_dataset = KineticsVideoDataset(mode='train', frame_num=CNN_FRAME_NUM, sampler_rate=sampler_rate, reduce_step=(4 if loss_index >= 4 else 0))
    else:
        train_dataset = VideoDataset(VIDEO_DIR, CNN_FRAME_NUM, sampler_rate, reduce_step=4 if loss_index >= 4 else 0)

    print(f'Training set contains {len(train_dataset)} videos')
    print(f'Batch size is {batch_size*WORLD_SIZE}')

    mp.spawn(main, args=(train_dataset,), nprocs=WORLD_SIZE)
