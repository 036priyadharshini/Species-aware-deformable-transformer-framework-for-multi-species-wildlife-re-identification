import os
import random
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cdist as scipy_cdist
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

CHECKPOINT_PATH = None                  # set to resume from a checkpoint
EDGE_CACHE_DIR  = './edge_cache'
DATA_ROOT       = './data/wildlifereid-10k'
METADATA_FILE   = os.path.join(DATA_ROOT, 'metadata.csv')
SAVE_DIR        = './checkpoints'

# 9-species configuration (thesis run).
# For the 7-species paper configuration, remove 'seaturtle' and 'faropig'.
SPECIES_LIST = [
    'tiger', 'cat', 'giraffe', 'cow', 'dog',
    'leopard', 'whale', 'seaturtle', 'faropig'
]
SPECIES_TO_IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
NUM_SPECIES    = len(SPECIES_LIST)

EPOCHS     = 350
P, K       = 8, 4          # PK sampler: P identities, K images each
EMBED_DIM  = 2048
D_MODEL    = 256
IMAGE_SIZE = 256
NUM_IMAGES = 4
MS_WEIGHT  = 1.0           # weight of Multi-Similarity loss

if torch.cuda.is_available():
    _cap   = torch.cuda.get_device_capability(0)
    device = torch.device('cuda' if _cap[0] >= 7 else 'cpu')
else:
    device = torch.device('cpu')

print(f"Device: {device}")


# =============================================================================
# Data augmentation
# =============================================================================

train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply(
        [T.RandomAffine(degrees=90, translate=(0.3, 0.3), scale=(0.8, 1.3))], p=0.5
    ),
    T.RandomPerspective(distortion_scale=0.5, p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tta_transforms = [
    T.Compose([
        T.Resize((256, 256)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    T.Compose([
        T.Resize((256, 256)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    T.Compose([
        T.Resize((288, 288)), T.CenterCrop(256), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    T.Compose([
        T.Resize((288, 288)), T.CenterCrop(256), T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    T.Compose([
        T.Resize((256, 256)),
        T.ColorJitter(brightness=0.15, contrast=0.15), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    T.Compose([
        T.Resize((256, 256)), T.RandomHorizontalFlip(p=1.0),
        T.ColorJitter(brightness=0.15, contrast=0.15), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


# =============================================================================
# Edge cache utilities
# =============================================================================

def _npy_path(cache_dir: str, img_relative_path: str) -> Path:
    h = hashlib.md5(img_relative_path.encode()).hexdigest()
    return Path(cache_dir) / f"{h}.npy"


def load_edge(cache_dir: str, cache_key: str, image_size: int = 256) -> np.ndarray:
    """Load a cached edge map, or return a zero array if not cached."""
    npy = _npy_path(cache_dir, cache_key)
    if npy.exists():
        return np.load(npy).astype(np.float32) / 255.0
    return np.zeros((image_size, image_size), dtype=np.float32)


# =============================================================================
# Model components
# =============================================================================

class ReDeformTRBackbone(nn.Module):
    """Lightweight 7-layer CNN backbone producing a 3-level feature pyramid."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.layer1 = self._make_layer(in_channels, 64,  stride=2)
        self.layer2 = self._make_layer(64,          128, stride=2)
        self.layer3 = self._make_layer(128,         256, stride=2)
        self.layer4 = self._make_layer(256,         256, stride=2)
        self.layer5 = self._make_layer(256,         256, stride=2)
        self.layer6 = self._make_layer(256,         256, stride=1)
        self.layer7 = self._make_layer(256,         256, stride=1)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x  = self.layer1(x)
        x  = self.layer2(x)
        x  = self.layer3(x)
        x  = self.layer4(x)
        x5 = self.layer5(x)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        return [
            x5,
            F.adaptive_avg_pool2d(x6, 4),
            F.adaptive_avg_pool2d(x7, 2),
        ]


class DeformableAttention(nn.Module):
    """
    Simplified deformable attention over a multi-scale feature pyramid.
    Follows the formulation in Zhu et al. (Deformable DETR, ICLR 2021).
    """

    def __init__(
        self,
        d_model:  int = 256,
        n_heads:  int = 8,
        n_levels: int = 3,
        n_points: int = 4,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets  = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj  = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias,   0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias,   0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query: torch.Tensor, pyramid: list) -> torch.Tensor:
        B, Nq, C = query.shape
        M, N, L  = self.n_heads, self.n_points, self.n_levels

        offsets = self.sampling_offsets(query).view(B, Nq, M, L, N, 2)
        attn_w  = self.attention_weights(query).view(B, Nq, M, L, N)
        attn_w  = F.softmax(attn_w.flatten(2), dim=-1).view(B, Nq, M, L, N)

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, 1, 8, device=query.device),
            torch.linspace(0, 1, 8, device=query.device),
            indexing='ij',
        )
        ref_pts = torch.stack([ref_x, ref_y], dim=-1).reshape(Nq, 2)

        all_feats, all_attn = [], []
        for li, feat in enumerate(pyramid):
            locs = ref_pts.view(1, Nq, 1, 1, 2) + offsets[:, :, :, li]
            grid = (locs * 2 - 1).reshape(B, Nq * M * N, 1, 2)
            sf   = F.grid_sample(
                feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False
            )
            sf = sf.squeeze(-1).view(B, C, Nq, M, N).permute(0, 2, 3, 4, 1)
            all_feats.append(sf)
            all_attn.append(attn_w[:, :, :, li])

        sf = torch.stack(all_feats, 3).reshape(B, Nq, M, L * N, C)
        aw = torch.stack(all_attn,  3).reshape(B, Nq, M, L * N, 1)
        return self.output_proj((sf * aw).sum(3).sum(2))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model:  int = 256,
        n_heads:  int = 8,
        n_levels: int = 3,
        ffn_dim:  int = 1024,
    ):
        super().__init__()
        self.self_attn = DeformableAttention(d_model, n_heads, n_levels)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(ffn_dim, d_model), nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, pyramid: list) -> torch.Tensor:
        q = self.norm1(q + self.self_attn(q, pyramid))
        q = self.norm2(q + self.ffn(q))
        return q


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (Deng et al., CVPR 2019).
    Adds margin in angle space for tighter inter-class boundaries.
    """

    def __init__(
        self,
        embed_dim:   int   = 2048,
        num_classes: int   = 2624,
        s:           float = 64.0,
        m:           float = 0.5,
    ):
        super().__init__()
        self.s           = s
        self.m           = m
        self.num_classes = num_classes
        self.weight      = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine  = F.linear(emb, F.normalize(self.weight, p=2, dim=1))
        cosine  = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta   = torch.acos(cosine)
        one_hot = F.one_hot(labels, self.num_classes).float()
        target  = torch.cos(theta + self.m * one_hot)
        return F.cross_entropy(self.s * target, labels)


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss (Wang et al., CVPR 2019).
    Optimises all pairwise relationships in a batch via three weighted terms
    (self-similarity, negative-similarity, positive-similarity), with adaptive
    hard-pair mining. Correlates directly with the AP metric.

    Standard hyperparameters: alpha=2, beta=50, base=0.5, eps=0.1.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta:  float = 50.0,
        base:  float = 0.5,
        eps:   float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.base  = base
        self.eps   = eps

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sim   = torch.mm(emb, emb.t())   # [B, B] cosine similarity (emb is L2-normalised)
        B     = emb.shape[0]
        loss  = torch.tensor(0.0, device=emb.device, requires_grad=True)
        count = 0

        for i in range(B):
            pos_mask    = labels == labels[i]
            neg_mask    = ~pos_mask
            pos_mask[i] = False

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_sims = sim[i][pos_mask]
            neg_sims = sim[i][neg_mask]

            pos_thresh = neg_sims.max().detach() + self.eps
            neg_thresh = pos_sims.min().detach() - self.eps

            pos_sims = pos_sims[pos_sims < pos_thresh]
            neg_sims = neg_sims[neg_sims > neg_thresh]

            if pos_sims.numel() == 0 or neg_sims.numel() == 0:
                continue

            pos_loss = (1.0 / self.alpha) * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (pos_sims - self.base)))
            )
            neg_loss = (1.0 / self.beta) * torch.log(
                1 + torch.sum(torch.exp(self.beta * (neg_sims - self.base)))
            )

            loss  = loss + pos_loss + neg_loss
            count += 1

        return loss / max(count, 1)


class UniversalReDeformTRV5(nn.Module):
    """
    Species-Aware Deformable Transformer for Wildlife Re-Identification.

    Key design choices:
    - 4-channel input (RGB + edge map) to capture structural markings.
    - Species embedding conditioning on transformer query tokens.
    - ArcFace + Multi-Similarity hybrid loss for discriminative embeddings.
    - 2048-dim L2-normalised output embedding for cosine retrieval.
    """

    def __init__(
        self,
        num_images:       int = 4,
        d_model:          int = 256,
        embed_dim:        int = 2048,
        n_encoder_layers: int = 4,
        n_heads:          int = 8,
        n_levels:         int = 3,
        ffn_dim:          int = 1024,
        num_queries:      int = 64,
        num_classes:      int = 2624,
        num_species:      int = 9,
    ):
        super().__init__()
        self.num_images = num_images
        self.backbone   = ReDeformTRBackbone(in_channels=4)
        self.query_init = nn.Linear(256, d_model)

        self.species_embed = nn.Embedding(num_species, d_model)
        nn.init.normal_(self.species_embed.weight, std=0.02)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, n_levels, ffn_dim)
            for _ in range(n_encoder_layers)
        ])

        self.embedding_head = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.BatchNorm1d(d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(d_model * 4, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        self.arcface = ArcFaceLoss(embed_dim, num_classes)
        self.ms_loss = MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)

    def forward(
        self,
        x:          torch.Tensor,
        species_id: torch.Tensor = None,
    ) -> torch.Tensor:
        B, I, C, H, W = x.shape
        pyramid = self.backbone(x.view(B * I, C, H, W))
        q       = self.query_init(pyramid[0].flatten(2).transpose(1, 2))

        if species_id is not None:
            sp_emb = self.species_embed(species_id)
            sp_emb = sp_emb.unsqueeze(1).unsqueeze(1)
            sp_emb = sp_emb.expand(B, I, q.shape[1], -1)
            sp_emb = sp_emb.reshape(B * I, q.shape[1], -1)
            q      = q + sp_emb

        for layer in self.encoder_layers:
            q = layer(q, pyramid)

        feat = q.mean(1).view(B, I, -1).mean(1)
        return F.normalize(self.embedding_head(feat), p=2, dim=1)


# =============================================================================
# Dataset
# =============================================================================

class WildlifeDataset(Dataset):
    """
    WildlifeReID-10K dataset loader with seeded identity sampling.
    Seeded shuffle ensures num_classes is deterministic across runs,
    preventing ArcFace head size mismatches when resuming training.
    """

    def __init__(
        self,
        data_root:       str,
        metadata_file:   str,
        edge_cache_dir:  str,
        species_list:    list,
        split:           str   = 'train',
        max_per_species: int   = 300,
        num_images:      int   = 4,
        transform=None,
        seed:            int   = 42,
    ):
        self.data_root      = Path(data_root)
        self.edge_cache_dir = edge_cache_dir
        self.num_images     = num_images
        self.transform      = transform or train_transform
        self.species_to_idx = {sp: i for i, sp in enumerate(species_list)}

        meta = pd.read_csv(metadata_file, low_memory=False)
        meta = meta[meta['species'].isin(species_list)]
        meta = meta[meta['split'] == split]

        id_imgs    = defaultdict(list)
        id_species = {}
        for _, row in meta.iterrows():
            iid  = str(row['identity'])
            path = self.data_root / row['path']
            if path.exists():
                id_imgs[iid].append((path, str(row['path'])))
                id_species[iid] = row['species']

        sp_ids = defaultdict(list)
        for iid in id_imgs:
            sp_ids[id_species[iid]].append(iid)

        rng          = random.Random(seed)
        selected_ids = []
        for sp, ids in sp_ids.items():
            ids_sorted = sorted(ids)
            rng.shuffle(ids_sorted)
            selected_ids.extend(ids_sorted[:max_per_species])

        self.samples         = []
        self.labels          = []
        self.species_ids     = []
        self.identity_to_idx = {}

        for iid in selected_ids:
            imgs = id_imgs[iid]
            if len(imgs) < 2:
                continue
            if iid not in self.identity_to_idx:
                self.identity_to_idx[iid] = len(self.identity_to_idx)
            self.samples.append(imgs)
            self.labels.append(self.identity_to_idx[iid])
            self.species_ids.append(self.species_to_idx[id_species[iid]])

        print(f"  {split}: {len(self.identity_to_idx)} identities | {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        imgs       = self.samples[idx]
        label      = self.labels[idx]
        species_id = self.species_ids[idx]
        chosen     = random.choices(imgs, k=self.num_images)

        batch = []
        for img_path, cache_key in chosen:
            try:
                img  = Image.open(img_path).convert('RGB')
                rgb  = self.transform(img)
                edge = torch.from_numpy(
                    load_edge(self.edge_cache_dir, cache_key)
                ).unsqueeze(0)
                batch.append(torch.cat([rgb, edge], dim=0))
            except Exception:
                batch.append(torch.zeros(4, IMAGE_SIZE, IMAGE_SIZE))

        return (
            torch.stack(batch),
            torch.tensor(label,      dtype=torch.long),
            torch.tensor(species_id, dtype=torch.long),
        )


# =============================================================================
# PK Sampler
# =============================================================================

class PKSampler(Sampler):
    """
    PK batch sampler: each batch contains P identities with K images each.
    Ensures effective hard-pair mining for the Multi-Similarity Loss.
    """

    def __init__(self, dataset: WildlifeDataset, P: int = 8, K: int = 4):
        self.dataset      = dataset
        self.P            = P
        self.K            = K
        self.label_to_idx = defaultdict(list)
        for i, lbl in enumerate(dataset.labels):
            self.label_to_idx[lbl].append(i)

    def __iter__(self):
        labels  = list(self.label_to_idx.keys())
        random.shuffle(labels)
        batches = []
        for i in range(0, len(labels) - self.P + 1, self.P):
            batch = []
            for lbl in labels[i: i + self.P]:
                idxs = self.label_to_idx[lbl]
                batch.extend(random.choices(idxs, k=self.K))
            batches.append(batch)
        random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        return (len(self.label_to_idx) // self.P) * self.P * self.K


# =============================================================================
# Evaluation helpers
# =============================================================================

def build_test_samples(
    data_root:      str,
    metadata_file:  str,
    species_list:   list,
    edge_cache_dir: str,
    seed:           int = 42,
):
    meta = pd.read_csv(metadata_file, low_memory=False)
    meta = meta[meta['species'].isin(species_list)]
    meta = meta[meta['split'] == 'test']

    id_imgs    = defaultdict(list)
    id_species = {}
    for _, row in meta.iterrows():
        iid  = str(row['identity'])
        path = Path(data_root) / row['path']
        if path.exists():
            id_imgs[iid].append((path, str(row['path'])))
            id_species[iid] = row['species']

    random.seed(seed)
    query_data, gallery_data = [], []
    identity_to_idx          = {}

    for iid, imgs in id_imgs.items():
        if len(imgs) < 4:
            continue
        if iid not in identity_to_idx:
            identity_to_idx[iid] = len(identity_to_idx)
        shuf   = imgs.copy()
        random.shuffle(shuf)
        sp_idx = SPECIES_TO_IDX.get(id_species[iid], 0)
        query_data.append({
            'imgs': shuf[:2], 'label': identity_to_idx[iid],
            'species': id_species[iid], 'species_id': sp_idx,
        })
        gallery_data.append({
            'imgs': shuf[2:4], 'label': identity_to_idx[iid],
            'species': id_species[iid], 'species_id': sp_idx,
        })

    print(f"  Query: {len(query_data)} | Gallery: {len(gallery_data)}")
    return query_data, gallery_data


def extract_features_tta(
    model:          nn.Module,
    samples:        list,
    edge_cache_dir: str,
    desc:           str = "TTA",
):
    feats, labels, species_out, species_ids_out = [], [], [], []
    model.eval()
    with torch.no_grad():
        for s in tqdm(samples, desc=desc):
            img_paths = s['imgs']
            while len(img_paths) < NUM_IMAGES:
                img_paths = img_paths + img_paths
            img_paths = img_paths[:NUM_IMAGES]

            sp_tensor  = torch.tensor([s['species_id']], dtype=torch.long).to(device)
            view_feats = []

            for tfm in tta_transforms:
                batch = []
                for img_path, cache_key in img_paths:
                    try:
                        img  = Image.open(img_path).convert('RGB')
                        rgb  = tfm(img)
                        edge = torch.from_numpy(
                            load_edge(edge_cache_dir, cache_key)
                        ).unsqueeze(0)
                        batch.append(torch.cat([rgb, edge], dim=0))
                    except Exception:
                        continue
                if len(batch) == NUM_IMAGES:
                    inp  = torch.stack(batch).unsqueeze(0).to(device)
                    feat = model(inp, sp_tensor)
                    view_feats.append(feat.cpu())

            if view_feats:
                avg = torch.stack(view_feats).mean(0)
                avg = F.normalize(avg, p=2, dim=1)
                feats.append(avg)
                labels.append(s['label'])
                species_out.append(s['species'])
                species_ids_out.append(s['species_id'])

    return (
        torch.cat(feats),
        np.array(labels),
        np.array(species_out),
        np.array(species_ids_out),
    )


# =============================================================================
# Re-ranking
# =============================================================================

def k_reciprocal_reranking_core(
    q_feat:       torch.Tensor,
    g_feat:       torch.Tensor,
    k1:           int   = 20,
    k2:           int   = 6,
    lambda_value: float = 0.5,
) -> np.ndarray:
    """
    K-reciprocal re-ranking (Zhong et al., CVPR 2017) with expanded
    neighbourhood (k1=20) tuned for the V5 embedding space.
    """
    Q       = F.normalize(q_feat, p=2, dim=1).numpy()
    G       = F.normalize(g_feat, p=2, dim=1).numpy()
    feat    = np.concatenate([Q, G], axis=0)
    n_query = Q.shape[0]
    n_all   = feat.shape[0]

    original_dist  = scipy_cdist(feat, feat, metric='euclidean').astype(np.float32)
    original_dist /= (original_dist.max() + 1e-12)
    initial_rank   = np.argsort(original_dist, axis=1)

    V = np.zeros((n_all, n_all), dtype=np.float32)
    for i in range(n_all):
        fwd   = initial_rank[i, : k1 + 1]
        bwd   = initial_rank[fwd, : k1 + 1]
        fi    = np.where(bwd == i)[0]
        recip = fwd[fi // (k1 + 1)] if fi.size else np.array([], dtype=np.int64)

        recip_exp = recip.copy()
        for cand in recip:
            hk   = int(np.around(k1 / 2))
            cf   = initial_rank[cand, : hk + 1]
            cb   = initial_rank[cf, : hk + 1]
            cfi  = np.where(cb == cand)[0]
            cr   = cf[cfi // (hk + 1)] if cfi.size else np.array([], dtype=np.int64)
            if len(np.intersect1d(cr, recip)) > 2 / 3 * len(cr):
                recip_exp = np.union1d(recip_exp, cr)

        recip_exp = recip_exp.astype(np.int64)
        if recip_exp.size:
            w = np.exp(-original_dist[i, recip_exp])
            V[i, recip_exp] = w / (w.sum() + 1e-12)

    if k2 > 1:
        V_qe = np.zeros_like(V)
        for i in range(n_all):
            V_qe[i] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe

    inv_index = [np.where(V[:, j] != 0)[0] for j in range(n_all)]
    n_gallery  = G.shape[0]
    jaccard    = np.zeros((n_query, n_gallery), dtype=np.float32)

    for i in range(n_query):
        nz_idx = np.where(V[i] != 0)[0]
        temp   = np.zeros(n_all, dtype=np.float32)
        for idx in nz_idx:
            rows = inv_index[idx]
            temp[rows] += np.minimum(V[i, idx], V[rows, idx])
        jaccard[i] = 1 - temp[n_query:] / (2 - temp[n_query:] + 1e-12)

    return lambda_value * original_dist[:n_query, n_query:] + (1 - lambda_value) * jaccard


def species_aware_reranking(
    q_feat:   torch.Tensor,
    g_feat:   torch.Tensor,
    q_species: np.ndarray,
    g_species: np.ndarray,
    k1:           int   = 20,
    k2:           int   = 6,
    lambda_value: float = 0.5,
) -> np.ndarray:
    """
    Species-aware re-ranking: applies k-reciprocal re-ranking within each
    species subset, avoiding cross-species neighbourhood contamination.
    """
    final_dist = torch.cdist(q_feat, g_feat).numpy().astype(np.float32)

    for sp in np.unique(q_species):
        q_idx = np.where(q_species == sp)[0]
        g_idx = np.where(g_species == sp)[0]
        if len(q_idx) < 4 or len(g_idx) < 4:
            continue
        print(f"    Re-ranking {sp}: {len(q_idx)}q x {len(g_idx)}g")
        block = k_reciprocal_reranking_core(
            q_feat[q_idx], g_feat[g_idx], k1=k1, k2=k2, lambda_value=lambda_value
        )
        final_dist[np.ix_(q_idx, g_idx)] = block

    return final_dist


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(dist_matrix, q_labels, g_labels, q_species_arr):
    cmc     = np.zeros(10)
    ap_list = []
    sp_aps  = defaultdict(list)
    sp_cmc  = defaultdict(lambda: np.zeros(10))
    sp_cnt  = defaultdict(int)

    for i in range(len(q_labels)):
        idx     = np.argsort(dist_matrix[i])
        matches = g_labels[idx] == q_labels[i]
        for k in range(10):
            if matches[:k + 1].any():
                cmc[k] += 1
                sp_cmc[q_species_arr[i]][k] += 1
        sp_cnt[q_species_arr[i]] += 1
        if matches.any():
            pos = np.where(matches)[0]
            ap  = np.mean([(j + 1) / (p + 1) for j, p in enumerate(pos)])
            ap_list.append(ap)
            sp_aps[q_species_arr[i]].append(ap)

    n    = len(q_labels)
    cmc /= n
    mAP  = np.mean(ap_list) if ap_list else 0.0
    sp_map = {sp: np.mean(v) for sp, v in sp_aps.items()}
    sp_r1  = {sp: sp_cmc[sp][0] / sp_cnt[sp] for sp in sp_cmc}
    return cmc, mAP, sp_map, sp_r1


def print_results(label, cmc, mAP, sp_map, sp_r1):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  mAP:    {mAP:.2%}")
    print(f"  Rank-1: {cmc[0]:.2%}  Rank-5: {cmc[4]:.2%}  Rank-10: {cmc[9]:.2%}")
    print(f"\n  Per-Species:")
    for sp in sorted(sp_map):
        bar = '#' * int(sp_map[sp] * 15)
        print(f"    {sp:15}: mAP {sp_map[sp]:.1%} | R1 {sp_r1.get(sp, 0):.1%} {bar}")


def full_evaluation(model, data_root, metadata_file, species_list, edge_cache_dir, epoch):
    print(f"\n{'=' * 60}")
    print(f"  Evaluation @ epoch {epoch}")
    print(f"{'=' * 60}")

    query_data, gallery_data = build_test_samples(
        data_root, metadata_file, species_list, edge_cache_dir
    )
    q_feat, q_lbl, q_sp, _ = extract_features_tta(
        model, query_data, edge_cache_dir, "Query  (TTA)"
    )
    g_feat, g_lbl, g_sp, _ = extract_features_tta(
        model, gallery_data, edge_cache_dir, "Gallery(TTA)"
    )

    dist_cos = torch.cdist(q_feat, g_feat).numpy()
    cmc, mAP, sp_map, sp_r1 = compute_metrics(dist_cos, q_lbl, g_lbl, q_sp)
    print_results(f"TTA cosine (epoch {epoch})", cmc, mAP, sp_map, sp_r1)

    print("\n  Running species-aware re-ranking...")
    dist_rr = species_aware_reranking(q_feat, g_feat, q_sp, g_sp)
    cmc_rr, mAP_rr, sp_map_rr, sp_r1_rr = compute_metrics(dist_rr, q_lbl, g_lbl, q_sp)
    print_results(f"TTA + Species-Aware Re-ranking (epoch {epoch})",
                  cmc_rr, mAP_rr, sp_map_rr, sp_r1_rr)

    best_mAP = max(mAP, mAP_rr)
    print(f"\n  Best mAP this eval: {best_mAP:.2%}")
    return best_mAP


# =============================================================================
# Training loop
# =============================================================================

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(EDGE_CACHE_DIR, exist_ok=True)

    print("Loading train dataset...")
    train_ds = WildlifeDataset(
        DATA_ROOT, METADATA_FILE, EDGE_CACHE_DIR,
        SPECIES_LIST, split='train', max_per_species=300,
        num_images=NUM_IMAGES, transform=train_transform, seed=42,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=PKSampler(train_ds, P=P, K=K),
        num_workers=4, pin_memory=True,
    )

    num_classes = len(train_ds.identity_to_idx)
    print(f"num_classes = {num_classes}")

    # Probe checkpoint for num_classes to avoid ArcFace head size mismatch.
    if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
        ckpt_probe      = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        model_num_classes = ckpt_probe.get('num_classes', num_classes)
        print(f"Checkpoint num_classes: {model_num_classes} | Dataset: {num_classes}")
    else:
        model_num_classes = num_classes

    model = UniversalReDeformTRV5(
        num_classes=model_num_classes,
        num_species=NUM_SPECIES,
        embed_dim=EMBED_DIM,
        d_model=D_MODEL,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.2f}M")

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(),       'lr': 5e-5,  'weight_decay': 1e-4},
        {'params': model.species_embed.parameters(),  'lr': 1e-4,  'weight_decay': 1e-4},
        {'params': model.query_init.parameters(),     'lr': 1e-4,  'weight_decay': 1e-4},
        {'params': model.encoder_layers.parameters(), 'lr': 1e-4,  'weight_decay': 1e-4},
        {'params': model.embedding_head.parameters(), 'lr': 2e-4,  'weight_decay': 1e-4},
        {'params': model.arcface.parameters(),        'lr': 2e-4,  'weight_decay': 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )

    start_epoch = 0
    best_map    = 0.0

    if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
        print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
        ckpt        = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        state       = ckpt['model_state_dict']
        model_state = model.state_dict()

        matched, skipped = {}, []
        for k, v in state.items():
            new_k = k.replace('cosface.', 'arcface.') if k.startswith('cosface.') else k
            new_k = new_k.replace('triplet.', 'ms_loss.') if new_k.startswith('triplet.') else new_k
            if new_k in model_state and model_state[new_k].shape == v.shape:
                matched[new_k] = v
            else:
                skipped.append(k)

        model_state.update(matched)
        model.load_state_dict(model_state)

        start_epoch = ckpt.get('epoch', 0)
        best_map    = ckpt.get('mAP', 0.0)
        print(f"  Resumed epoch {start_epoch} | best mAP {best_map:.2%}")
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")
    else:
        print("Training from scratch.")

    print(f"\nStarting training: epoch {start_epoch + 1} -> {EPOCHS}")
    print("=" * 60)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        total_loss = total_arc = total_ms = 0.0
        n_batches  = 0

        for imgs, labels, species_ids in train_loader:
            imgs        = imgs.to(device)
            labels      = labels.to(device)
            species_ids = species_ids.to(device)

            optimizer.zero_grad()
            emb      = model(imgs, species_ids)
            arc_loss = model.arcface(emb, labels)
            ms_loss  = model.ms_loss(emb, labels)
            loss     = arc_loss + MS_WEIGHT * ms_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_arc  += arc_loss.item()
            total_ms   += ms_loss.item()
            n_batches  += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        avg_arc  = total_arc  / max(n_batches, 1)
        avg_ms   = total_ms   / max(n_batches, 1)
        lr_now   = optimizer.param_groups[-1]['lr']

        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"\nEpoch [{epoch}/{EPOCHS}] LR: {lr_now:.2e} | "
                  f"Loss: {avg_loss:.4f} (arc: {avg_arc:.4f} ms: {avg_ms:.4f})")
            mAP = full_evaluation(
                model, DATA_ROOT, METADATA_FILE, SPECIES_LIST, EDGE_CACHE_DIR, epoch
            )

            save_path = Path(SAVE_DIR) / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mAP':                  mAP,
                'num_classes':          model_num_classes,
                'num_species':          NUM_SPECIES,
                'species_list':         SPECIES_LIST,
                'embed_dim':            EMBED_DIM,
            }, save_path)
            print(f"  Saved: {save_path}")

            if mAP > best_map:
                best_map  = mAP
                best_path = Path(SAVE_DIR) / 'best_model.pth'
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'mAP':              mAP,
                    'num_classes':      model_num_classes,
                    'num_species':      NUM_SPECIES,
                    'species_list':     SPECIES_LIST,
                    'embed_dim':        EMBED_DIM,
                }, best_path)
                print(f"  New best: {best_map:.2%} -> {best_path}")
        else:
            print(f"Epoch [{epoch}/{EPOCHS}] Loss: {avg_loss:.4f} "
                  f"(arc: {avg_arc:.4f} ms: {avg_ms:.4f})")

    print(f"\n{'=' * 60}")
    print(f"  Training complete. Best mAP: {best_map:.2%}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
