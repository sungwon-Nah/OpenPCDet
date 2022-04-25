## Study note
### train.py
- parse_config():

 > cfg_file : model define 
 >> cfg.SOMETHING is defined in various kitti_models/##.yaml file.
 
 > batch_size : batch size for training
 > epochs : 
 > ckpt : pre-trained model
 
 - main() :
 
 > train_set, train_loader, train_sampler = **build_dataloader**(
 dataset_config,
 class_names,
 batch_size,
 dist,
 workers,
 logger,
 training=T/F,
 merge_all_iters_to_one_epoch,
 total_epoch) ,
 
> ***model*** = **build_network**(model, class, dataset)
> model.cuda()
 
> optimizer = **build_optimizer**(model, cfg.optimization)

> model.train()
>> if dist_train() --> DataParallel for usage of multi-gpus

> lr_schedulaer, lr_warmup_scheduler = **build_scheduler**(optimizer, 
opotal_iters_each_epoch, total_epochs, last_epoch, optimzation_cfg)

> **train_model**(model, 
optimizer,
train_loader,
model_func(?),
lr_scheduler,
optimizer_cfg,
start_epoch,
total_epoch,
start_iter,
rank,
tb_log,
ckpt_save_dir,
train_sampler,
lr_warmup_scheduler,
ckpt_save_interval,
max_ckpt_save_num,
merge_all_iters_to_one_epoch
)

Train is over here. 
After here, Evaluation started.

To summarize, we should dig into 5 methods.
`build_dataloader`
`build_network`
`build_optimizer`
`build_scheduler`
`train_model`

---

* **build_dataloader**  (defined in `pcdet/datasets/__init__.py`)
Basically, it uses `DataLoader from torch.utils.data`
there is Template for each dataset
```
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset
```
As a result, if you want to utilize custom dataset, you have to define a custom dataset in the datasets `root/pcdet/datasets` directory.

Basically, DatasetTemplate inherits torch_data.Dataset.
As a result, defined `dataset` type is torch_data.Dataset. 
torch.nn.DataLoader(`dataset`, batch_size, pin_memory, num_workers, shuffle, collate_fn, drop_last, sampler, timeout)

If you want to train model on your custom dataset, refer to dataset tamplate.

---

 * **build_network** (defined in `pcdet/models/__init__.py`)
its input is `model_cfg`, `num_class`, `dataset` , and return is `model`.
main function here is **build_detector**( `model_cfg`, `num_class`, `dataset`).
As a result, let's dig into **build_detector**.
 
 * **build_detector** (defined in `pcdet/models/detectors/__init__.py`)
Similiar to datase_builder, there are template for detecdtor model.
```
__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus
}
``` 
Basically, `Detector3DTemplate` inherits  `nn.Module`.
Therefore,  if you want to develop new deep learning model, define your own model with `nn.Module` type.
As for variant `Detector3DTemplate` existing, let's look into **PVRCNN** model.


**PVRCNN(Detector3DTemplate)**
 it is common concept of deep learning model, which consists of `nn.Module`. 
 it has `forward()` and `get_training_loss()` methods.
 Most of function is inherited by **Detector3DTemplate**.
 
 **Detector3DTemplate**
 This is literally the key point of studying models.
 Let's list up the functions in the `Detector3DTemplate`.
 
- update_global_step()
- build_networks()
- build_vfe()
- build_backbone_3d()
- build_map_to_bev_module()
- build_backbone_2d()
- build_pfe()
- build_dense_head()
- build_point_head()
- build_roi_head()
- forward()
- post_processing()
- generate_recall_record()
- load_state_dict()
- load_params_from_file()
- load_params_with_optimizer()

**build_network()** in the Detector3DTemplate class.
All the variables defined in the `build_network()` comes from kitti_dataset.yaml file.
It define the network charateristics for datasets.

`module_list` : 
```
self.module_topology = [
    'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
    'backbone_2d', 'dense_head',  'point_head', 'roi_head'
]
...
for module_name in self.module_topology:
    module, model_info_dict = getattr(self, 'build_%s' % module_name)(
        model_info_dict=model_info_dict
    )
    self.add_module(module_name, module)
```
`num_rawpoint_features` : from .processor.point_feature_encoder import PointFeatureEncoder
`num_point_feature` : from .processor.point_feature_encoder import PointFeatureEncoder
```
 POINT_FEATURE_ENCODING: {
    encoding_type: absolute_coordinates_encoding,
    used_feature_list: ['x', 'y', 'z', 'intensity'],
    src_feature_list: ['x', 'y', 'z', 'intensity'],
}
```


 `grid_size` : 
`voxel_size`
 `depth_downsample_factor`.
```
DATA_PROCESSOR:
    - NAME: mask_points_and_boxes_outside_range
      REMOVE_OUTSIDE_BOXES: True

    - NAME: shuffle_points
      SHUFFLE_ENABLED: {
        'train': True,
        'test': False
      }

    - NAME: transform_points_to_voxels
      VOXEL_SIZE: [0.05, 0.05, 0.1]
      MAX_POINTS_PER_VOXEL: 5
      MAX_NUMBER_OF_VOXELS: {
        'train': 16000,
        'test': 40000
      }

``` 

 `point_cloud_range` :
 ```
 POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
 ```
 
 
---



