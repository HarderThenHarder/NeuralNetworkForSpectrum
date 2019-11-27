"""
@author: P_k_y
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


class DataVision:

    def __init__(self):
        self.spec_tsne_2d = None
        self.spec_tsne_3d = None
        self.font = {"color": 'w', "size": 13, "family": "serif"}
        plt.style.use("dark_background")

    def plot_vision_result(self, data, target):
        """
        Plot the Result Image of 2D and 3D
        :param data: Features data
        :param target: Class Index
        :return: None
        """
        total_class = len(set(list(target)))
        print("\n-> Generating the vision result, this may take a while...")
        self.spec_tsne_2d = TSNE(n_components=2).fit_transform(data)
        self.spec_tsne_3d = TSNE(n_components=3).fit_transform(data)

        fig = plt.figure(figsize=(12.5, 4))

        # plot 2D View
        ax2d = fig.add_subplot(121)
        ax0 = ax2d.scatter(self.spec_tsne_2d[:, 0], self.spec_tsne_2d[:, 1], c=target, alpha=0.6, cmap=plt.cm.get_cmap("rainbow", total_class))
        plt.title("Data Distribution in 2D", fontdict=self.font)
        cbar = fig.colorbar(ax0, ax=ax2d)
        cbar.set_label(label="Material ID", fontdict=self.font)
        ax2d.grid(True, linestyle='--', alpha=0.3)

        # Plot 3D View
        ax3d = fig.add_subplot(122, projection='3d')
        ax1 = ax3d.scatter(self.spec_tsne_3d[:, 0], self.spec_tsne_3d[:, 1], self.spec_tsne_3d[:, 2], c=target, alpha=0.6, cmap=plt.cm.get_cmap("rainbow", total_class))
        plt.title("Data Distribution in 3D", fontdict=self.font)
        cbar = fig.colorbar(ax1, ax=ax3d)
        cbar.set_label(label="Material ID", fontdict=self.font)

        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("data/T_Original.csv", header=None)
    spec_target = data.values[:, 0]
    spec_data = data.values[:, 1:]
    dv = DataVision()
    dv.plot_vision_result(spec_data, spec_target)
