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
$$ build network() \supset build detector() \supset Detector3DTemplate()$$

All the variables defined in the `build_network()` comes from kitti_dataset.yaml file and each model yaml file(ex. pv_rcnn.yaml).
From the kitti_dataset.yaml file, It define the network charateristics for datasets.

`module_list` : It builds the network model. 
There are module_topology dictionary for convenience.
```
self.module_topology = [
    'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
    'backbone_2d', 'dense_head',  'point_head', 'roi_head'
]
```
If you see the whole lines in the `detector3d_template.py`, you can find a method starting with build_xx.
For example,
```
- build_vfe()
- build_backbone_3d()
- build_map_to_bev_module()
- build_backbone_2d()
- build_pfe()
- build_dense_head()
- build_point_head()
- build_roi_head()
```
These methods generate module of model following configure file (`pv_rcnn.yaml`).
```
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
 
For your understanding, let's summarize model structure defined in the `pcdet/model`.

```
MODEL:
    VFE:
		-VFETemplate
		-MeanVFE
		-PillarVFE
		-ImageVFE
		-DynMeanVFE
		-DynPillarVFE    

    BACKBONE_3D:
    	-VoxelBackBone8x
		-UNetV2
		-PointNet2Backbone
		-PointNet2MSG
		-VoxelResBackBone8x

    MAP_TO_BEV:
    	-HeightCompression
		-PointPillarScatter
		-Conv2DCollapse

    BACKBONE_2D:
		-BaseBEVBackbone
    
    DENSE_HEAD:
		-AnchorHeadTemplate
		-AnchorHeadSingle
		-PointIntraPartOffsetHead
		-PointHeadSimple
		-PointHeadBox
		-AnchorHeadMulti
		-CenterHead
    
    PFE:
		-VoxelSetAbstraction
		
    POINT_HEAD:
		-AnchorHeadTemplate
		-AnchorHeadSingle
		-PointIntraPartOffsetHead
		-PointHeadSimple
		-PointHeadBox
		-AnchorHeadMulti
		-CenterHead
    
    ROI_HEAD:
		-RoIHeadTemplate
		-PartA2FCHead
		-PVRCNNHead
		-SECONDHead
		-PointRCNNHead
		-VoxelRCNNHead
		
    POST_PROCESSING:
	    	...
```
All the model can be found at the `pcdet/models` directory.
Pre-defined model topologies can be adopted easily.

**NOTE**
Generally, `BACKBONE` is used as a feature extractor, which gives you a feature map representation of the input.
`HEAD` is used as a performing the actual task, such as detection, segmentation, etc. 
This way is similar to a head attached to the backbone.

Let's see more detail for each module and its parameters.

```
VFE : Voxel Feature Extraction 
	- num_rawpoint_features
	- point_cloud_range
	- voxel_size
	- grid_size
	- depth_downsample_factor

BACKBONE_3D 
	- num_point_features
	- grid_size
	- voxel_size
	- point_cloud_range
	- backbone_channels
	
MAP_TO_BEV
	- grid_size
	- num_bev_features

BACKBONE_2D
	- num_bev_features --> output of MAP_TO_BEV
	
PFE : Point Feature Extraction
	- 
	...

```
---

* **build_optimizer()** is defined at the `tools/train_utils/optimization/__init__.py`
We can set the configuration of the optimization at the end of the *model.yaml* file. 
```
optim_cfg.OPTIMIZER == 'adam' or 'sgd' or 'adam_onecycle'
```
We can set the batch size per gpu and number of epochs ... etc.

```
OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 2
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.01
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10
```

---

* **build_scheduler()** is defined at the `tools/train_utils/optimization/__init__.py` as like **build_optimizer()**.
The output of `scheduler()` is *lr_scheduler*, *lr_warmup_scheduler*. 
As a result, the output is determined by `build_optimzier()`. 
Except for `adam_onecycle` optimizer, `adam` and `sgd` is scheduled by `LambdaLR`.
`adam_onecycle` uses `OneCycle` to determine a *lr_scheduler*.
```
    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

```

---

We have learned about **train.py** code.
To summarize this study note, your contribution can be different depending on your applications.

1. If you want to train on your custom dataset, you should study dataset configurations, and parameters for build_networks.
Also, build_dataloader().

2. If you want to make a new model for the detecting 3D objects, please study about build_network and lots of modules defined in the model yaml file.




