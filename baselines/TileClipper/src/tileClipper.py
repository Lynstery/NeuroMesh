############################################################################
# TileClipper: Lightweight Selection of Regions of Interest from Videos for 
# Traffic Surveillance                                                      
# Copyright (C) 2024 Shubham Chaudhary, Aryan Taneja, Anjali Singh,         
# Purbasha Roy, Sohum Sikdar, Mukulika Maity, Arani Bhattacharya            
                                                                            
# This program is free software: you can redistribute it and/or modify      
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
###########################################################################


#################################################################################################
# This scipt has main source code for TileClipper which uses cluster based strategy to filter 
# unwanted tiles. It requires GPAC's MP4Box for tile manipulation. 
# Assuming that the calibration phase is already done separately, run TileClipper as below:
#   python tileClipper.py --tiled-video-dir dataset/tiled_4x4_mp4/videoName 
#                         --percentile-array-filename ../assets/F2s/f2s_videoName__cluster10.pkl 
#                         --cluster-indices-file ../assets/F2s/videoName_cluster_indices.pkl
#################################################################################################

from __future__ import annotations
from pathlib import Path
import numpy as np
import subprocess as sp
import time, argparse
from tqdm import tqdm
import os
import joblib as jb

class EvictingQueue():
    '''
    Class to implement single ended queue/evicting
    queue. It only provides method to add with no
    dequeue method. The element from the other
    end gets removed automatically on adding
    an element if the queue is full.
    '''
    def __init__(self, elements: list|tuple, size: int = 10) -> None:
        self.size = size
        if len(elements) == size:
            self.__list = [i for i in elements]
        elif len(elements) > size:
            self.__list = [elements[(-size + i)] for i in range(size)]
        else:
            self.__list = [i for i in elements]


    def append(self, element: float|int) -> None:
        """
        Adds elements to end of the queue and removes one 
        element from front if queue size > specified one.
        """
        if type(element) == float or type(element) == int:
            self.__list.append(element)
        else:
            raise TypeError("Elements should of type float or int")
        if len(self.__list) > self.size:
            self.__list.pop(0)


    def __repr__(self) -> str:
        return f"EvictingQueue([{', '.join([str(i) for i in self.__list])}])"


    def __len__(self) -> int:
        return len(self.__list)


    def __iter__(self):
        return iter(self.__list)


class TileClipper():
    """
    Implements TileClipper that removes tiles without any objects
    in a tiled video based on the statistics of past few segments' bitrates. 
    """
    def __init__(self, static_tiles: set = {2}, total_tiles: int = 16, number_of_calibration_segments: int = 60, object_ratio_limit: float = 0.1, cluster_size: int = 10, gamma: float = 1.75):
        self.staticTiles = static_tiles # Tile 2 cannot be removed because of codec constraint.
        self.totalTiles = total_tiles
        self.numberOfCalibrationSegments = number_of_calibration_segments
        self.clusterSize = cluster_size
        self.gamma = gamma
        self.objectRatioLimit = object_ratio_limit

    def removeTiles(self, video_path, target_path, list_of_tiles_to_remove: list):                # e.g. list = [1,2,5,6,8]
        lst = [str(list_of_tiles_to_remove[(i//2)-1]) if(i%2==0) else "-rem" for i in range(1,2*len(list_of_tiles_to_remove)+1)]
        sp.run(["MP4Box"] + lst + [video_path, "-quiet", "-out", target_path], stdout = sp.DEVNULL, stderr = sp.DEVNULL)

    def copyFile(self, video_path, target_path) -> None:
        sp.run(["cp", video_path, target_path], stdout = sp.DEVNULL, stderr = sp.DEVNULL)

    def readBestPercentileFileFromServer(self, file_name: str):
        """
        Reads the best percentile file generated during calibration.
        """
        return np.array(jb.load(file_name))

    def readClusterIndicesFileFromServer(self, file_name: str) -> tuple:
        """
        Reads the file from server to get the indices of bitrates of 
        both the clusters got during calibration.
        """
        return jb.load(file_name)

    def getBestPercentileForClustersOfATile(self, percentile_array: np.ndarray, tile_num: int) -> list:
        """
        Returns the best cluster percentile to use for a tile using the 
        values got during calibration.
        """
        tmp = np.where(percentile_array[:, 2] == tile_num)[0]
        return percentile_array[tmp][np.argmax(percentile_array[tmp, 3])][:2]

    def getSelectedTilesForGaussianScheme(self, true_cluster: list, false_cluster: list, bitrate: float, lower_percentile: int, upper_percentile: int) -> tuple[bool, int]:
        """
        Selects tile using cluster based classification with
        outlier detection (gamma*sigma) for tiles with obj_ratio < 0.1
        """
        if false_cluster != None:
            currentThreshold = (np.percentile(list(true_cluster), lower_percentile) + np.percentile(list(false_cluster), upper_percentile)) / 2
        else:
            currentThreshold = np.median(list(true_cluster)) + (self.gamma * np.std(list(true_cluster)))
            true_cluster.append(bitrate)

        selected = False
        if bitrate > currentThreshold:
            selected = True
            if false_cluster != None:
                true_cluster.append(bitrate)
        else:
            if false_cluster != None:
                false_cluster.append(bitrate)
        return selected, currentThreshold

    def getClustersUsingGaussianForNoObject(self, first_n_true_cluster_indxs: np.ndarray, first_n_false_cluster_indxs: np.ndarray, bitrates_during_calibration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns true and false clusters. Each cluster is a single ended queue.
        """
        if (len(first_n_true_cluster_indxs) / self.numberOfCalibrationSegments) < self.objectRatioLimit:
            _trueCluster = EvictingQueue(list(bitrates_during_calibration), size = self.numberOfCalibrationSegments)
            _falseCluster = None
        else:
            _trueCluster = EvictingQueue(list(bitrates_during_calibration[first_n_true_cluster_indxs]), size = self.clusterSize)
            if len(first_n_false_cluster_indxs) != 0:
                _falseCluster = EvictingQueue(list(bitrates_during_calibration[first_n_false_cluster_indxs]), size = self.clusterSize)
            else:
                _falseCluster = EvictingQueue([min(bitrates_during_calibration[first_n_true_cluster_indxs])], size = self.clusterSize)
   
        return _trueCluster, _falseCluster
    
    def prepare(self, tiled_path, cluster_indices_file_from_server, percentile_array_file_from_server) -> np.ndarray:
        clusterIndicesList = self.readClusterIndicesFileFromServer(cluster_indices_file_from_server)
        bestPercentileArray = self.readBestPercentileFileFromServer(percentile_array_file_from_server)
        bestPercentileForClusters = []

        for tile_indx in range(self.totalTiles):
            l, u = self.getBestPercentileForClustersOfATile(bestPercentileArray, tile_indx)
            bestPercentileForClusters.append([l, u]) 

        bitrates = np.zeros((self.numberOfCalibrationSegments, self.totalTiles))
        for i, data in enumerate(sorted(Path(tiled_path).iterdir())): 
            bitrate = sp.run(["ffprobe", "-v", "error",
                        "-show_entries", "stream=bit_rate",
                        "-of", "default=noprint_wrappers=1", 
                        data], stdout=sp.PIPE)

            arr = np.fromiter(map(lambda x: int(x[9:]), bitrate.stdout.decode().split('\n')[1:-1]), dtype=int) # 9 => 'bit_rate='; [1:-1] => 1 because tile 1 has metadata only
            if i < self.numberOfCalibrationSegments:
                bitrates[i] = arr
            else:
                break

        # Reading the percentile file got during calibration and finding best percentile to use for each tile
        _trueCluster, _falseCluster = [], []
        for tile_indx in range(self.totalTiles):   
            first_n_true_segments_indx, first_n_false_segments_indx = clusterIndicesList[tile_indx]
            trueCluster, falseCluster = self.getClustersUsingGaussianForNoObject(first_n_true_segments_indx, first_n_false_segments_indx, bitrates[:self.numberOfCalibrationSegments, tile_indx])
            _trueCluster.append(trueCluster)
            _falseCluster.append(falseCluster)
            
        self.bestPercentileForClusters = bestPercentileForClusters
        self._trueCluster = _trueCluster
        self._falseCluster = _falseCluster

    def run(self, seg_path, target_path) -> None:
        """
        Runs TileClipper on the tiled_video_segment_folder/.
        Uses the best percentile found using the file received
        from server (percentile_array_file_from_server and 
        cluster_indices_file_from_server)
        """
        bitrate = sp.run(["ffprobe", "-v", "error",
                    "-show_entries", "stream=bit_rate",
                    "-of", "default=noprint_wrappers=1", 
                    seg_path], stdout=sp.PIPE)
        arr = np.fromiter(map(lambda x: int(x[9:]), bitrate.stdout.decode().split('\n')[1:-1]), dtype=int)
        # 9 => 'bit_rate='; [1:-1] => 1 because tile 1 has metadata only

        tilesToRemove = []
        for _tile in range(self.totalTiles):
            selected, threshold = self.getSelectedTilesForGaussianScheme(true_cluster=self._trueCluster[_tile], false_cluster=self._falseCluster[_tile], bitrate=float(arr[_tile]), lower_percentile=self.bestPercentileForClusters[_tile][0], upper_percentile=self.bestPercentileForClusters[_tile][1])
            if selected == False:       
                tilesToRemove.append(_tile + 2) # +2 because GPAC tile indexing starts with 2

        # Checking values greater than threshold. And extracting their indices
        tmp = self.staticTiles.copy()
        tmp.update(tilesToRemove)
        tmp.discard(2) # Tile 2 cannot be removed due to codec constraint

        #print(f"remove: {list(tmp)}")

        # Removing unwanted tiles
        if len(tmp) == 0:
            self.copyFile(seg_path, target_path)
        else:
            self.removeTiles(seg_path, target_path, list(tmp))

if __name__ == "__main__":
    tileClipper = TileClipper()
    save_name = '0'
    imgs_dir = '/data/zh/videos_dir/0'
    tiled_path = os.path.join(imgs_dir, 'tiled') 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    percentile_array_filename = os.path.join(script_dir, f"../assets/F2s/f2s_{save_name}_cluster10.pkl")
    cluster_indices_file = os.path.join(script_dir, f"../assets/F2s/{save_name}_cluster_indices.pkl")
    tileClipper.prepare(tiled_path, cluster_indices_file, percentile_array_filename)
    
    tileClipper.run('/data/zh/videos_dir/0/tiled/seg0011.mp4', 'result.mp4')
    sp.run(["gpac", "-i", 'result.mp4', "tileagg", "@", "-o", 'result_agg.mp4']) 
    sp.run(["ffmpeg", "-i", 'result_agg.mp4', '-v', 'error', f"%04d.png"])