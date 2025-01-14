import matplotlib.pyplot as plt
import torch
from timm.models import create_model

from mesh.dataset.kinetics import KINETICS400_DIR
from mesh.dataset.utils import (DATA_DISK_DIR, ReshapeVideo,
                                timm_video_normalization)
from mesh.dnn_model.util import (convert_a_video_tensor_to_images,
                                 convert_a_video_to_images_and_tensor,
                                 convert_a_video_tensor_to_images_list,
                                 convert_tensor_to_an_image)
from mesh.msc.engine_for_pretraining import (generate_patchs_from_videos, 
                                             reconstruct_videos_from_patchs, get_video_patch_mean_std, reconstruct_videos_from_patchs_with_mean_std,
                                             video_frame_filtering)
from mesh.vitransformer.arguments import get_args
from mesh.vitransformer.pretrain import __all__

rank = 3
opts = get_args()
model = create_model(
    opts.model,
    pretrained=True,
    drop_path_rate=opts.drop_path,
    drop_block_rate=None,
    decoder_depth=opts.decoder_depth,
    mask_ratio=opts.mask_ratio,
    init_ckpt=DATA_DISK_DIR+'/mesh_model/1_vision/checkpoint-99.pth').to(rank)
model.eval()
patch_size = model.encoder.patch_embed.patch_size[0]


def show_videos_difference(frame_number, images, reconstructed_frames=None, select=None):
    '''
    Show image difference between original video and reconstructed video
    '''
    fig_size = 3
    fig_width = frame_number*fig_size
    fig_height = fig_size
    if reconstructed_frames is not None:
        fig_height *= 2
    fig = plt.figure(figsize=(fig_width, fig_height))

    for i in range(frame_number):
        num = i+1
        ax = fig.add_subplot(2, frame_number, num)
        ax.set_title('Frame'+str("{:02d}".format(num)))
        ax.imshow(images[i])
    if reconstructed_frames is not None:
        j = 0
        for i in range(frame_number):
            if (select is None) or (select is not None and select[i] == 1):
                num = i+1
                ax = fig.add_subplot(2, frame_number, frame_number + num)
                ax.set_title('Rec'+str("{:02d}".format(num)))
                ax.imshow(convert_tensor_to_an_image(
                    reconstructed_frames[j], False))
                j += 1

    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    plt.show()


def eval_image_reconstruction():
    imgs, inputs = convert_a_video_to_images_and_tensor(
        KINETICS400_DIR+'/k400_320p/ZZoxcS-rDGE_000288_000298.mp4',
        frame_num=opts.num_frames, video_tensor=True,
        new_size=[opts.input_size, opts.input_size], rank=rank)
    #imgs[0].show()
    imgs[0].save('/data/zh/original.png')
    with torch.no_grad():
        norm_inputs = timm_video_normalization(inputs)
        decoder_undergo_depth = 4
        outputs, p_x, bool_masked_pos = model(norm_inputs, decoder_undergo_depth)

        print(model.encoder.content_complexity)
        print(model.decoder.content_complexity)

        rec_videos = reconstruct_videos_from_patchs_with_mean_std(
            outputs, bool_masked_pos, opts.num_frames, opts.input_size, patch_size, mean, std)

        for i in range(0, opts.num_frames):
            img = convert_a_video_tensor_to_images(rec_videos[0], i, show=False)
            img.save(f'/data/zh/reconstructed_{i}.png')

        #rec_videos = reconstruct_videos_from_patchs(
        #    outputs, bool_masked_pos, opts.num_frames, opts.input_size, patch_size, inputs, False, True)
        #img = convert_a_video_tensor_to_images(rec_videos[0], 0, show=False)
        #img.save('/data/zh/reconstructed2.png')

        #rev = ReshapeVideo(opts.num_frames)
        #filtered_frames, select = video_frame_filtering(rev, rec_videos, p_x, patch_size)
        #print(select)

def eval_frame_filtering():
    imgs, inputs = convert_a_video_to_images_and_tensor(
        KINETICS400_DIR+'/k400_320p/ZZoxcS-rDGE_000288_000298.mp4',
        frame_num=opts.num_frames, video_tensor=True, frame_step=1,
        new_size=[opts.input_size, opts.input_size], rank=rank)

    with torch.no_grad():
        norm_inputs = timm_video_normalization(inputs)
        outputs, p_x, bool_masked_pos = model(norm_inputs)

        rec_videos = reconstruct_videos_from_patchs(
            outputs, bool_masked_pos, opts.num_frames, opts.input_size, patch_size, inputs, False, opts.normlize_target)
        rev = ReshapeVideo(opts.num_frames)
        video_frames, select = video_frame_filtering(
            rev, rec_videos, p_x, patch_size)

        print(select)
        show_videos_difference(opts.num_frames, imgs, video_frames, select)

def mesh_encode(inputs):
    # inputs: 1, 3, 16, 224, 224
    with torch.no_grad():
        original_patchs, mean, std = generate_patchs_from_videos(inputs, patch_size, True) 
        norm_inputs = timm_video_normalization(inputs)

        # encode
        x_vis, p_x, bool_masked_pos = model.encoder_forward(norm_inputs)
        visible_patchs = original_patchs[~bool_masked_pos]

    return x_vis, mean, std, p_x, bool_masked_pos, visible_patchs


def mesh_decode(x_vis, mean, std, p_x, bool_masked_pos, original_visible_patchs=None, decoder_undergo_depth=4):
    # decode
    outputs = model.decoder_forward(x_vis, bool_masked_pos, decoder_undergo_depth)

    if original_visible_patchs is not None:
        # use original visible patchs
        _, N = bool_masked_pos.shape
        masked_num = torch.sum(bool_masked_pos[0]).item()
        outputs[:, :(N-masked_num)] = original_visible_patchs

    rec_videos = reconstruct_videos_from_patchs_with_mean_std(
        outputs, bool_masked_pos, opts.num_frames, opts.input_size, patch_size, mean, std)

    return rec_videos

if __name__ == '__main__':
    # eval_image_reconstruction()
    # eval_frame_filtering()
    imgs, inputs = convert_a_video_to_images_and_tensor(
        #KINETICS400_DIR+'/k400_320p/ZZoxcS-rDGE_000288_000298.mp4',
        KINETICS400_DIR + '/k400_320p/fq6zxClFQxU_000332_000342.mp4',
        frame_num=opts.num_frames, video_tensor=True, frame_step=1,
        new_size=[opts.input_size, opts.input_size], rank=rank)

    x_vis, mean, std, p_x, bool_masked_pos, visible_patchs = mesh_encode(inputs)

    rec_videos = mesh_decode(x_vis, mean, std, p_x, bool_masked_pos, visible_patchs, 4)

    rev = ReshapeVideo(opts.num_frames)

    filtered_frames, select = video_frame_filtering(rev, rec_videos, p_x, patch_size)

    rec_frames = rev(rec_videos)
    for i in range(0, opts.num_frames):
        img = convert_tensor_to_an_image(rec_frames[i], show=False)
        img.save(f'/data/zh/tmp/rec_{i}.png')
        imgs[i].save(f'/data/zh/tmp/ori_{i}.png')

    print("select: ", select)
    #show_videos_difference(opts.num_frames, imgs, filtered_frames, select)