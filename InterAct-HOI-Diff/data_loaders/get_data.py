from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_behave_collate, t2m_contact_collate, t2m_omomo_collate, t2m_behave_bps_collate
from dataclasses import dataclass

def get_dataset_class(name):
    if name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "behave":
        from data_loaders.behave.data.dataset import Behave
        return Behave
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    num_frames: int
    split: str = 'train'
    hml_mode: str = 'train'
    training_stage: int = 1


def get_collate_fn(name, hml_mode='train', training_stage=1):
    if hml_mode == 'gt' and name in ["humanml", "kit"]:
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if hml_mode == 'gt' and name in ["behave"]:
        from data_loaders.behave.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if hml_mode == 'gt' and name in ["omomo"]:
        from data_loaders.omomo.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name in ["behave"] and training_stage==1:
        return t2m_contact_collate
    elif name in ["behave"] and training_stage==2:
        return t2m_behave_collate
    elif name in ["behave"] and training_stage==3:
        return t2m_behave_bps_collate
    else:
        return all_collate


def get_dataset(conf: DatasetConfig):
    DATA = get_dataset_class(conf.name)
    if conf.name in ["humanml",  "behave", "omomo"]:
        dataset = DATA(split=conf.split,
                       mode=conf.hml_mode,
                       num_frames=conf.num_frames,
                       training_stage=conf.training_stage)
    else:
        raise NotImplementedError()
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset



def get_dataset_loader(conf: DatasetConfig):
    # name, batch_size, num_frames, split='train', hml_mode='train'
    dataset = get_dataset(conf)
    collate = get_collate_fn(conf.name, conf.hml_mode, conf.training_stage)

    loader = DataLoader(
        dataset, batch_size=conf.batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate,
    )
    return loader