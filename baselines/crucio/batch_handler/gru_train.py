import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd.function import InplaceFunction
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from crucio.autoencoder.dataset import (GRU_FRAME_NUM, VIDEO_DIR, KINETICS400_DIR, VideoDataset, KineticsVideoDataset,
                                        preprocess_video_dataset)
from crucio.autoencoder.loss import loss_function
from crucio.autoencoder.gpu import WORLD_SIZE, ddp_setup, print_device_info
from crucio.batch_handler.gru_filter import (GRU_PATH,
                                             diversity_regularization,
                                             get_filter, normalize_scores,
                                             print_gru_info,
                                             representativeness_loss)

# Hyperparameter
use_kinetics_dataset = 1 
sampler_rate = 0.1
num_epochs = 60
save_epoch = 10
# True for probability (bernoulli) and False for threshold (dff_round)
probs_binary = False
# A larger rep_weight means a higher accuracy target
rep_weight = 0.5
loss_index = 2
if loss_index < 4:
    batch_size = 4
    learning_rate = WORLD_SIZE*5e-4
else:
    batch_size = 2
    learning_rate = WORLD_SIZE*5e-5


class dff_round(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        result = torch.round(input)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result


def main(rank, train_dataset):
    ddp_setup(rank)
    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(train_dataset), num_workers=8)
        if rank == 0:
            print(f'Length of training set (i.e. number of batch) is {len(train_loader)}')

        # Define model and optimizer
        # extractor, gru = get_filter(is_load=True, rank=rank)
        extractor, gru = get_filter(rank=rank, is_load=True)
        optimizer = optim.Adam(gru.parameters(), lr=learning_rate)
        # Multiply learning rate by 0.1 every 15 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        # Define loss function
        criterion = loss_function(loss_index, rank)

        # Training model
        for epoch in range(num_epochs):
            train_loss = 0
            train_loss_rep = 0
            train_loss_div = 0
            filter_ratio = 0
            train_acc = 0
            batch_index = 1
            train_loader.sampler.set_epoch(epoch)
            for i, (videos, _, paths) in enumerate(train_loader):
                optimizer.zero_grad()

                # Forward propagation
                videos = videos.to(rank)
                features = extractor(videos)
                features = (features - features.mean()) / features.std()
                scores = gru(features)

                if probs_binary:
                    # Sample probability for binary values
                    probability = Bernoulli(scores).probs
                    selects = torch.bernoulli(probability)
                else:
                    # Set threshold for binary values
                    scores = normalize_scores(scores, rank)
                    selects = dff_round.apply(scores)
                # Note: _batch_size<=batch_size

                _batch_size = selects.shape[0]
                # If no frame is selected, first frame is retained by default
                for _ in range(_batch_size):
                    if torch.all(selects[_] == 0):
                        selects[_][0] = 1
                # Calculate diversity regression
                div_loss = diversity_regularization(features, selects, rank)
                # Calculate representativeness loss
                rep_loss, rep_acc = representativeness_loss(criterion, loss_index, videos, selects, rank)
                # Calculate weighted loss function
                loss = (1-rep_weight) * div_loss + rep_weight * rep_loss
                #if probs_binary:
                    #loss *= (1-probability.mean())
                    #print(f"prob: {1-probability.mean()}\n")

                # Error propagation and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gru.parameters(),max_norm=5.0)

                #if rank == 0:
                    #for name, param in gru.named_parameters():
                    #    if param.grad is not None:
                    #        grad_norm = param.grad.norm()
                    #        grad_mean = param.grad.mean()
                    #        grad_std = param.grad.std()
                    #        print(f"Layer [{name}], Grad Norm [{grad_norm}], Mean [{grad_mean}], Std [{grad_std}]")

                optimizer.step()
                del videos
                torch.cuda.empty_cache()
                train_loss += loss.item()
                train_loss_rep += rep_loss.item()
                train_loss_div += div_loss.item()
                # Calculate filter ratio
                keyframe = 0
                for _ in range(_batch_size):
                    keyframe += torch.count_nonzero(selects[_])
                ratio = (GRU_FRAME_NUM-keyframe/_batch_size)/GRU_FRAME_NUM
                filter_ratio += ratio
                # Calculate accuracy
                if loss_index >= 4:
                    train_acc += rep_acc

                if rank == 0:
                    print(
                        f"Epoch [{epoch+1}] batch [{batch_index}/{len(train_loader)}]", end="\r")
                batch_index += 1

            # Automatically adjust learning rate
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            if rank == 0:
                if loss_index < 4:
                    print(
                        f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Loss: {train_loss/len(train_loader):.4f}, Loss Rep: {train_loss_rep/len(train_loader):.4f}, Loss Div: {train_loss_div/len(train_loader):.4f}, FR: {filter_ratio/len(train_loader):.4f}')
                else:
                    print(
                        f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Loss: {train_loss/len(train_loader):.4f}, Loss Rep: {train_loss_rep/len(train_loader):.4f}, Loss Div: {train_loss_div/len(train_loader):.4f}, FR: {filter_ratio/len(train_loader):.4f}, Acc: {train_acc/len(train_loader):.4f}')

            if (epoch+1) % save_epoch == 0:
                if rank == 0:
                    print(f"Epoch [{epoch+1}], saving trained weights")
                    torch.save(gru.module.state_dict(), GRU_PATH)

        if rank == 0:
            torch.save(gru.module.state_dict(), GRU_PATH)

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        destroy_process_group()


if __name__ == "__main__":
    print_device_info()
    print_gru_info()
    # Preprocess dataset
    # preprocess_video_dataset(VIDEO_DIR)
    # Prepare training dataset
    if use_kinetics_dataset:
        train_dataset = KineticsVideoDataset(mode='train', frame_num=GRU_FRAME_NUM, sampler_rate=sampler_rate, reduce_step=2, scale_ratio=1, frame_skip_step=10, frame_random_skip_step=False)
    else:
        train_dataset = VideoDataset(VIDEO_DIR, GRU_FRAME_NUM, sampler_rate, reduce_step=2)
    print(f'Training set contains {len(train_dataset)} videos')
    print(f'Batch size is {batch_size*WORLD_SIZE}')
    mp.spawn(main, args=(train_dataset,), nprocs=WORLD_SIZE)
