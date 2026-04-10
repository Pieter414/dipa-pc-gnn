from utils import sparse_to_adjlist
from scipy.io import loadmat
import argparse

"""
	Read data and save the adjacency matrices to adjacency lists
"""

def main(data_dir:str = "data/", data:str = "yelp"):
    if data == "yelp":
        yelp = loadmat(f"{data_dir}/YelpChi.mat")
        net_rur = yelp['net_rur']
        net_rtr = yelp['net_rtr']
        net_rsr = yelp['net_rsr']
        yelp_homo = yelp['homo']

        sparse_to_adjlist(net_rur, data_dir + 'yelp_rur_adjlists.pickle')
        sparse_to_adjlist(net_rtr, data_dir + 'yelp_rtr_adjlists.pickle')
        sparse_to_adjlist(net_rsr, data_dir + 'yelp_rsr_adjlists.pickle')
        sparse_to_adjlist(yelp_homo, data_dir + 'yelp_homo_adjlists.pickle')
    elif data == "amazon":
        amz = loadmat(f'{data_dir}/Amazon.mat')
        net_upu = amz['net_upu']
        net_usu = amz['net_usu']
        net_uvu = amz['net_uvu']
        amz_homo = amz['homo']

        sparse_to_adjlist(net_upu, data_dir + 'amz_upu_adjlists.pickle')
        sparse_to_adjlist(net_usu, data_dir + 'amz_usu_adjlists.pickle')
        sparse_to_adjlist(net_uvu, data_dir + 'amz_uvu_adjlists.pickle')
        sparse_to_adjlist(amz_homo, data_dir + 'amz_homo_adjlists.pickle')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN fraud detection experiment sweep")
    
    parser.add_argument("--data", type=str, default="yelp")
    parser.add_argument("--data-dir", type=str, default="data/")

    args = parser.parse_args()
    main(args.data_dir, args.data)

