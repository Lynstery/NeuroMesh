import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv3 import (MobileNet_V3_Small_Weights,
                                            mobilenet_v3_small)
from torchvision.models.squeezenet import (SqueezeNet1_0_Weights,
                                           SqueezeNet1_1_Weights,
                                           squeezenet1_0, squeezenet1_1)

from crucio.autoencoder.dataset import GRU_FRAME_NUM, MIN_FRAME_NUM
from crucio.autoencoder.util import CUDA_ENABLED, WEIGHTS_DIR

INPUT_SIZE = 1000
HIDDEN_SIZE = 256
LAYER_NUM = 1
GRU_PATH = WEIGHTS_DIR+'/weights_gru_' + \
    str(INPUT_SIZE)+'.'+str(HIDDEN_SIZE)+'.' + \
    str(LAYER_NUM)+'.'+str(GRU_FRAME_NUM)+'.pth'


def print_gru_info():
    print('Network parameters of GRU (Beneficial to keyframe extraction but longer filtering time)')
    print(
        f'INPUT_SIZE={INPUT_SIZE} (Must be same size as VideoExtractor output)')
    print(f'HIDDEN_SIZE={HIDDEN_SIZE}')
    print(f'LAYER_NUM={LAYER_NUM}')
    print(f'GRU_FRAME_NUM={GRU_FRAME_NUM}')
    print(f'GRU_PATH={GRU_PATH}')


class VideoExtractor(nn.Module):
    def __init__(self):
        super(VideoExtractor, self).__init__()
        self.extractor = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.extractor = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.extractor = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        self.extractor = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[1], shape[-2], shape[-1])
        x = self.extractor(x)
        features = x.reshape(shape[0], shape[2], -1)
        size = x.shape[-1]
        return features


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, LAYER_NUM,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*HIDDEN_SIZE, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        self.gru.flatten_parameters()
        features, _ = self.gru(x)
        scores = self.fc(features).squeeze()
        scores = scores.reshape(x.shape[0], -1)
        scores = self.sig(scores)
        return scores


def get_filter(mode='train', is_load=False, rank=0):
    '''
    mode -> Set network mode to 'train' or 'eval'
    is_load -> Whether to load GRU model weights 
               for continued training or evaluation
    '''
    extractor = VideoExtractor()
    gru = GRUModel()
    if is_load:
        gru.load_state_dict(torch.load(GRU_PATH, weights_only=False))
    if mode == 'train':
        extractor = extractor.to(rank)
        # extractor = DDP(extractor)
        gru = gru.to(rank)
        gru = DDP(gru)
        gru.train()
    elif mode == 'eval':
        if CUDA_ENABLED:
            extractor = extractor.to(rank)
            gru = gru.to(rank)
        gru.eval()
    extractor.eval()
    return extractor, gru


def diversity_regularization(features, selects, rank=0, zero=1e-4):
    '''
    Calculate diversity regression of current selection
    '''
    assert features.shape[0] == selects.shape[0]
    batch_size = features.shape[0]
    dpp_loss = 0
    for _ in range(batch_size):
        N = selects[_]
        assert torch.all(N == 0) == False
        X = features[_]
        # Similarity matrix L is calculated based on Gaussian kernel function
        sigma = 1
        L = torch.exp(-torch.cdist(X, X, p=2) ** 2 / (2 * sigma ** 2)).to(rank)
        # Compute similarity of selected subset
        # selected = torch.nonzero(N).squeeze()
        # L_s = L.index_select(0, selected).index_select(1, selected)
        selected = torch.count_nonzero(N)
        mask = N.reshape(-1, 1)*N
        masked_L = torch.mul(L, mask)
        L_s = torch.masked_select(masked_L, mask == 1).reshape(selected, selected)
        det_L_s = torch.det(L_s)
        assert det_L_s >= 0
        # Calculate DPP probability
        I = torch.eye(X.size(0)).to(rank)
        det_L = torch.det(L + I)
        dpp = (det_L_s+zero) / det_L
        assert dpp <= 1
        prob = -torch.log(dpp)
        #print(f"prob:\n {prob.item()}")
        assert prob >= 0
        dpp_loss += prob
    dpp_loss /= batch_size
    return dpp_loss


def calculate_filtered_accuracy(acc_mat, select, rank=0):
    '''
    Element (column j of row i) in matrix acc_mat represent accuracy of frame i with frame j as Ground Truth
    '''
    frame_num = select.shape[0]
    frame_acc = torch.zeros(frame_num).to(rank)
    for _ in range(frame_num):
        if select[_] == 1:
            frame_acc[_] = 1
        else:
            left_indices = (select[:_] == 1).nonzero()
            if left_indices.numel() != 0:
                resue_frame = left_indices.max()
            else:
                right_indices = (select[_+1:] == 1).nonzero()
                if right_indices.numel() != 0:
                    resue_frame = right_indices.min()
                else:
                    frame_acc[_] = 0
            frame_acc[_] = acc_mat[_][resue_frame]
    return frame_acc.mean()


def representativeness_loss(criterion, loss_index, videos, selects, rank=0, inf=1e5):
    '''
    Representativeness loss of current selection is calculated based on loss function criterion
    '''
    assert videos.shape[0] == selects.shape[0]
    batch_size = videos.shape[0]
    rep_loss = 0
    rep_acc = 0
    for _ in range(batch_size):
        N = selects[_]
        assert torch.all(N == 0) == False
        X = videos[_]
        X = X.transpose(0, 1)
        number = N.shape[0]
        rep_mat = torch.zeros(number, number).to(rank)
        if loss_index >= 4:
            acc_mat = torch.zeros(number, number).to(rank)
        for i in range(number):
            for j in range(number):
                if N[j] == 1:
                    # TODO: Calculate loss of frame combinations in parallel
                    with torch.no_grad():
                        if loss_index < 4:
                            loss = criterion(
                                X[i].unsqueeze(0), X[j].unsqueeze(0))
                        else:
                            loss, acc = criterion(
                                X[i].unsqueeze(0), X[j].unsqueeze(0))
                            acc_mat[i][j] = acc
                        rep_mat[i][j] = loss
        # indices = torch.nonzero(N).squeeze()
        # rep_mat = rep_mat[:, indices].reshape(number, -1)
        indices = inf*(torch.ones(N.shape[0], requires_grad=True).to(rank)-N)
        rep_mat += indices.unsqueeze(0)
        rep_mat = torch.exp(rep_mat.min(dim=1, keepdim=True)[0])
        rep_loss += rep_mat.mean()
        if loss_index >= 4:
            rep_acc += calculate_filtered_accuracy(acc_mat, N)
    rep_loss /= batch_size
    rep_acc /= batch_size
    return rep_loss, rep_acc


def normalize_scores(scores, rank=0):
    batch_size = scores.shape[0]
    normalized_scores = torch.zeros(scores.shape, requires_grad=True).to(rank)
    for _ in range(batch_size):
        score = scores[_]
        normalized_scores[_] = (score - score.min()) / \
            (score.max() - score.min())
    return normalized_scores


def scores_to_selects(scores):
    scores = normalize_scores(scores)
    selects = torch.round(scores)
    # Adjust selects according to MIN_FRAME_NUM
    batch_size = scores.shape[0]
    for _ in range(batch_size):
        ones = (selects[_] == 1).sum().item()
        if ones < MIN_FRAME_NUM:
            count = MIN_FRAME_NUM - ones
            indices = torch.where(scores[_] < 0.5)[0]
            top_indices = torch.topk(
                scores[_][indices], k=count).indices
            result_indices = indices[top_indices]
            selects[_][result_indices] = 1
    return selects


def apply_select_to_video(selects, videos):
    length = len(selects)
    selects = torch.nonzero(selects)
    select_videos = []
    for _ in range(length):
        idx = (selects[:, 0] == _).nonzero().squeeze()
        select = selects[idx, 1]
        video = videos[_]
        select_video = torch.index_select(video, dim=1, index=select)
    select_videos.append(select_video)
    return select_videos


def test_diversity_regularization(rank=0):
    feature = torch.tensor(
        [[[1, 2, 3], [1, 2, 3], [2, 3, 4], [5, 6, 7]],
         [[3, 2, 1], [3, 2, 1], [3, 4, 5], [3, 4, 5]]],
        dtype=torch.float32).to(rank)
    selects1 = torch.ones(2, 4, dtype=torch.float32).to(rank)
    selects2 = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 1]],
                            dtype=torch.float32).to(rank)
    selects3 = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0]],
                            dtype=torch.float32).to(rank)
    print(selects1)
    print(diversity_regularization(feature, selects1))
    print(selects2)
    print(diversity_regularization(feature, selects2))
    print(selects3)
    print(diversity_regularization(feature, selects3))


if __name__ == '__main__':
    test_diversity_regularization()
