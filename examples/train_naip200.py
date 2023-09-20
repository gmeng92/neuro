# main file for training the meshnet and unet using naip200 data
# libraries and modules
import os
import sys
import numpy as np
importa pandas as pd

import torch
import nibabel as nib
# TODO: add the performace metrics from the catalyst library
from catalyst.metrics.functional._segmentation import dice
import matplotlib.pyplot as plt

from neuro.predictor import Predictor
from neuro.model import MeshNet, UNet


# Specify the parameters
volume_shape = [256, 256, 256]
subvolume_shape = [64, 64, 64]
n_subvolumes = 32
n_classes = 4
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

#Case 1: Using the pretrained model parameters from the 
#TODO: add the pretrained model loader

#Case 2: Fine-tuning based on the pretrained model parameters

#Case 3: Training from scratch
#Data loader for the naip200 dataset
def get_loaders(
  random_state: int = 123,
  volume_shape: List[int],
  subvolume_shape: List[int],
  in_csv_train: str = None,
  in_csv_valid: str = None,
  in_csv_infer: str = None,
  batch_size: int = 32,
  # remark: num_workers=0 is important or it may raise an error 
  num_workers: int = 0,
) -> dict:

  datasets = {}
  # global open_fn
  open_fn = ReaderCompose(
      [
          NiftiFixedVolumeReader(input_key="images", output_key="images"),
          NiftiReader(input_key="nii_labels", output_key="targets"),

      ]
  )

  for mode, source in zip(("train", "validation", "infer"),
                          (in_csv_train, in_csv_valid, in_csv_infer)):
      if mode == "infer":
          n_subvolumes = 128#512
      else:
          n_subvolumes = 32#128

      if source is not None and len(source) > 0:
          dataset = BrainDataset(
              list_data=dataframe_to_list(pd.read_csv(source)),
              list_shape=volume_shape,
              list_sub_shape=subvolume_shape,
              open_fn=open_fn,
              n_subvolumes=n_subvolumes,
              mode=mode,
              input_key="images",
              output_key="targets",
          )

      datasets[mode] = {"dataset": dataset}
      print(#dataset.data, 
            dataset.mode, 
            dataset.subjects,
            dataset.n_subvolumes, 
            dataset.dict_transform,
            dataset.open_fn,
            dataset.input_key,
            dataset.output_key,
            dataset.subvolume_shape)
  global worker_init_fn
  def worker_init_fn(worker_id):
      np.random.seed(np.random.get_state()[1][0] + worker_id)


  train_loader = DataLoader(dataset=datasets['train']['dataset'], 
                            batch_size=batch_size,
                            shuffle=True, 
                            worker_init_fn=worker_init_fn,
                            num_workers=0, 
                            pin_memory=True)
  valid_loader = DataLoader(dataset=datasets['validation']['dataset'],
                            batch_size=batch_size,
                            shuffle=True, 
                            worker_init_fn=worker_init_fn,
                            num_workers=0, 
                            pin_memory=True,
                            drop_last=True)
  test_loader = DataLoader(dataset=datasets['infer']['dataset'],
                           batch_size=batch_size,
                           worker_init_fn=worker_init_fn,
                           num_workers=0, 
                           pin_memory=True,
                           drop_last=True)
  # debug
  print(train_loader)
  train_loaders = collections.OrderedDict()
  infer_loaders = collections.OrderedDict()
  train_loaders["train"] = BatchPrefetchLoaderWrapper(train_loader)
  train_loaders["valid"] = BatchPrefetchLoaderWrapper(valid_loader)
  infer_loaders['infer'] = BatchPrefetchLoaderWrapper(test_loader)

  return train_loaders, infer_loaders
# The training dependencies
class CustomRunner(Runner):

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def predict_batch(self, batch):
        # model inference step
        batch = batch[0]
        return self.model(batch['images'].float().to(self.device)), batch['coords']

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
          key: metrics.AdditiveValueMetric(compute_on_call=False)
          for key in ["loss", "macro_dice"]
        }

    def handle_batch(self, batch):
        # model train/valid step
        batch = batch[0]
        x, y = batch['images'].float(), batch['targets']

        if self.is_train_loader:
          self.optimizer.zero_grad()

        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        if self.is_train_loader:
          loss.backward()
          self.optimizer.step()
          scheduler.step()

        one_hot_targets = (
          torch.nn.functional.one_hot(y, num_classes=4)
          .permute(0, 4, 1, 2, 3)
          .cuda()
        )

        logits_softmax = F.softmax(y_hat)
        macro_dice = dice(logits_softmax, one_hot_targets, mode='macro')

        self.batch_metrics.update({"loss": loss, 'macro_dice': macro_dice})
        for key in ["loss", "macro_dice"]:
          self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)

    def on_loader_end(self, runner):
        for key in ["loss", "macro_dice"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
                                           
# The training loop

train_loaders, infer_loaders = get_loaders(
    123, volume_shape, subvolume_shape,
    "./data/dataset_train.csv",
    "./data/dataset_valid.csv",
    "./data/dataset_infer.csv")

n_classes = 4
n_epochs = 50
meshnet = MeshNet(n_channels=1, n_classes=n_classes)
unet = UNet(n_channels=1, n_classes=n_classes)

logdir = "logs/naip200"

optimizer = torch.optim.Adam(meshnet.parameters(), lr=0.02)

scheduler = OneCycleLR(optimizer, max_lr=.02,
                 epochs=n_epochs, steps_per_epoch=len(train_loaders['train']))

runner = CustomRunner()
runner.train(
    model=meshnet,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=train_loaders,
    num_epochs=n_epochs,
    logdir=logdir,
    callbacks=[CheckpointCallback(logdir=logdir)],
    verbose=True
)
# The performance meature

# majority voting for subvolumes
def voxel_majority_predict_from_subvolumes(loader, n_classes, segmentations):
    if segmentations is None:
        for subject in range(loader.dataset.subjects):
            segmentations[subject] = torch.zeros(
                tuple(np.insert(loader.volume_shape, 0, n_classes)),
                dtype=torch.uint8).cpu()

    prediction_n = 0
    for inference in tqdm(runner.predict_loader(loader=loader)):
        coords = inference[1].cpu()
        _, predicted = torch.max(F.log_softmax(inference[0].cpu(), dim=1), 1)
        for j in range(predicted.shape[0]):
            c_j = coords[j][0]
            subj_id = prediction_n // loader.dataset.n_subvolumes
            for c in range(n_classes):
                segmentations[subj_id][c, c_j[0, 0]:c_j[0, 1],
                                       c_j[1, 0]:c_j[1, 1],
                                       c_j[2, 0]:c_j[2, 1]] += (predicted[j] == c)
            prediction_n += 1

    for i in segmentations.keys():
        segmentations[i] = torch.max(segmentations[i], 0)[1]
    return segmentations

segmentations = {}
for subject in range(infer_loaders['infer'].dataset.subjects):
    segmentations[subject] = torch.zeros(tuple(np.insert(volume_shape, 0, n_classes)), dtype=torch.uint8)

# The inference results
segmentations = voxel_majority_predict_from_subvolumes(infer_loaders['infer'],
                                                     n_classes, segmentations)
subject_metrics = []
for subject, subject_data in enumerate(tqdm(infer_loaders['infer'].dataset.data)):
    # Ground truth labels
    seg_labels = nib.load(subject_data['labels']).get_fdata()
    segmentation_labels = torch.nn.functional.one_hot(
        torch.from_numpy(seg_labels).to(torch.int64), n_classes)

    inference_dice = dice(
        torch.nn.functional.one_hot(segmentations[subject], n_classes).permute(0, 3, 1, 2),
        segmentation_labels.permute(0, 3, 1, 2)
    ).detach().numpy()
    macro_inference_dice = dice(
        torch.nn.functional.one_hot(segmentations[subject], n_classes).permute(0, 3, 1, 2),
        segmentation_labels.permute(0, 3, 1, 2), mode='macro'
    ).detach().numpy()
    subject_metrics.append((inference_dice, macro_inference_dice))

# classwise and macro dice
per_class_df = pd.DataFrame([metric[0] for metric in subject_metrics])
macro_df = pd.DataFrame([metric[1] for metric in subject_metrics])
print(per_class_df, macro_df)
print(macro_df.mean())