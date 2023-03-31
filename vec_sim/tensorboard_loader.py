import os
from coco_to_milvus import *
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import clip
import pickle
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import random




BATCH_SIZE = 100
SERIALIZE_DIR = 'tensorboard/serialized_val_embs/'
LOG_DIR = 'tensorboard/logs'
IMG_PATH = 'tensorboard/serialized_val_embs/imgs/'
CAPTION_FILE = ANN_FILE_FMT.format(DATA_DIR, 'val2017')


def serialize():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device {device}')
    model, preprocess = clip.load(CLIP_MODEL, device)
    caption_file = ANN_FILE_FMT.format(DATA_DIR, 'val2017')
    coco_caps = COCO(caption_file)
    all_img_ids_capfile = coco_caps.getImgIds() # integer list

    N = len(all_img_ids_capfile)
    all_img_ids_capfile.sort()
    
    # LOOP THROUGH BATCHES OF IMAGES:
    start_index = 0

    img_embeddings = np.zeros((N, EMB_DIM))
    cap_embeddings = []
    img_ids_all = []
    cap_ids_all = []

    print(f'{len(all_img_ids_capfile)} images to encode')
    for batch in chunker(all_img_ids_capfile, BATCH_SIZE):
        start_time = time.time()
        batch_img, img_ids, img_emb, cap_ids, cap_emb = embed_coco_img_batch(coco_caps, batch, 'val2017', model, preprocess, device, True)
        print(f'encoding time {time.time() - start_time}')
        bsize = len(batch)
        print(cap_emb.shape)
        img_embeddings[start_index:(start_index + bsize),:] = img_emb
        cap_embeddings.append(cap_emb)
    
        img_ids_all.append(img_ids)
        cap_ids_all.append(cap_ids)

        np.save(SERIALIZE_DIR+'imgs/'+str(start_index)+'imgs.npy', batch_img)
        print(f'sent {start_index}-{start_index+bsize-1}')
        start_index += bsize
    cap_embeddings = np.concatenate(cap_embeddings, 0)
    np.save(SERIALIZE_DIR+'img_embeddings.npy', img_embeddings)
    np.save(SERIALIZE_DIR+'cap_embeddings.npy', cap_embeddings)
    
    with open(SERIALIZE_DIR+'img_ids.pickle', 'wb') as handle:
        pickle.dump(img_ids_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(SERIALIZE_DIR+'cap_ids.pickle', 'wb') as handle:
        pickle.dump(cap_ids_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_img(writer):
    imgs = torch.Tensor(np.concatenate([np.load(IMG_PATH+x) for x in os.listdir(IMG_PATH)]))
    imgs = F.resize(imgs, [100,100], InterpolationMode.BILINEAR)
    print(imgs.shape)
    print('loaded serialized imgs')
    img_embeddings = torch.Tensor(np.load(SERIALIZE_DIR+'img_embeddings.npy'))
    print(img_embeddings.shape)
    print('adding embeddings...')
    writer.add_embedding(img_embeddings, label_img = imgs, tag='coco_img_val', global_step=0)
    print('done')
    
    
def load_caps(writer):
    cap_ids = None
    with open(SERIALIZE_DIR+'cap_ids.pickle', 'rb') as handle:
        cap_ids = pickle.load(handle)
    cap_ids =  [item for sublist in cap_ids for item in sublist][:10000]
    print(len(cap_ids))
    cap_embeddings = torch.Tensor(np.load(SERIALIZE_DIR+'cap_embeddings.npy'))[:10000]
    print(len(cap_embeddings))
    coco_caps = COCO(CAPTION_FILE)
    caption_jsons = coco_caps.loadAnns(cap_ids)
    captions = [x['caption'] for x in caption_jsons]
    writer.add_embedding(cap_embeddings, metadata=captions, tag='coco_cap_val', global_step=1)

    
def main():
    writer = SummaryWriter(log_dir=LOG_DIR)
    load_img(writer)
    load_caps(writer)
    writer.close()


if __name__ == "__main__":
    main()