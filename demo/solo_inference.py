""" rgb -> segmentation mask
"""
import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import argparse
import matplotlib.pyplot as plt
import os
from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
log_file = 'test.log'
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s - %(filename)s %(funcName)s %(asctime)s ", handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, help='csv')
parser.add_argument('--outfolder', type=str, help='output_folder')
parser.add_argument('--config_file', type=str, help='config file')
parser.add_argument('--checkpoint_file', type=str, help='checkpoint file')
params = parser.parse_args()

# config_file = '../configs/solo/decoupled_solo_r50_fpn_8gpu_3x.py'
# checkpoint_file = '../checkpoints/DECOUPLED_SOLO_R50_3x.pth'
config_file = params.config_file
checkpoint_file = params.checkpoint_file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

os.makedirs(params.outfolder, exist_ok=True)
root = os.path.dirname(params.csv)
df = pd.read_csv(params.csv)
# import pdb; pdb.set_trace()
for i, v in tqdm(df.iterrows(), total=len(df), desc="SOLO Segmentation"):
	try:
		img = join(root, df.at[i, 'rgb'])
		bname = os.path.splitext(os.path.basename(img))[0]
		ofname = join(params.outfolder, '{}_seg.png'.format(bname))
		if os.path.exists(ofname):
			df.at[i,'segmentation'] = os.path.relpath(ofname, root)
			continue

		result = inference_detector(model, img)
		seg = result[0][0][:1].detach().cpu().numpy().transpose((1,2,0)).repeat(3, axis=2).astype(np.float)
		plt.imsave(ofname, seg)
		df.at[i,'segmentation'] = os.path.relpath(ofname, root)
	except:
		logging.error("File {} has problem. Result: {}".format(img, result))


print("Finish solo inference")
df.to_csv(params.csv, index=False)
# show_result_ins(img, result, model.CLASSES, score_thr=0.25, out_file="demo_out.jpg")

