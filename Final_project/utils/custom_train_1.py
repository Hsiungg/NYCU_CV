import time
import os
import numpy as np
from cellpose import io, utils, models, dynamics, train
from cellpose.transforms import normalize_img, random_rotate_and_resize
from pathlib import Path
import torch
from torch import nn
from tqdm import trange
import logging

train_logger = logging.getLogger(__name__)

def train_seg_with_callback(net, train_data=None, train_labels=None, train_files=None,
              train_labels_files=None, train_probs=None, test_data=None,
              test_labels=None, test_files=None, test_labels_files=None,
              test_probs=None, channel_axis=None,
              load_files=True, batch_size=1, learning_rate=5e-5, SGD=False,
              n_epochs=100, weight_decay=0.1, normalize=True, compute_flows=False,
              save_path=None, save_every=100, save_each=False, nimg_per_epoch=None,
              nimg_test_per_epoch=None, rescale=False, scale_range=None, bsize=256,
              min_train_masks=5, model_name=None, class_weights=None, callback=None,
              validate_every=5, tb_writer=None,):
    """
    Extended version of train_seg that supports callbacks for monitoring training progress.
    
    Args:
        callback: A callback object with an on_epoch_end method that takes epoch, train_loss, and val_loss as arguments.
        validate_every: Run validation every N epochs. Default is 5.
        All other arguments are the same as in the original train_seg function.
    """
    if SGD:
        train_logger.warning("SGD is deprecated, using AdamW instead")

    device = net.device

    scale_range = 0.5 if scale_range is None else scale_range

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize
    
    out = train._process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_labels_files=train_labels_files,
                              train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_labels_files=test_labels_files,
                              test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channel_axis=channel_axis,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {"normalize_params": normalize_params, "channel_axis": channel_axis}
    
    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    if class_weights is not None and isinstance(class_weights, (list, np.ndarray, tuple)):
        class_weights = torch.from_numpy(class_weights).to(device).float()
        print(class_weights)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 99:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")
    train_logger.info(
        f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    filename = save_path / model_name / model_name
    (save_path / model_name).mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {filename}")

    lavg, nsum = 0, 0
    train_losses, test_losses = np.zeros(n_epochs), np.zeros(n_epochs)
    
    # Create progress bar for epochs
    epoch_pbar = trange(n_epochs, desc='Training', position=0, leave=True, ncols=100)
    for iepoch in epoch_pbar:
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]
        net.train()
        
        # Create progress bar for training batches
        train_pbar = trange(0, nimg_per_epoch, batch_size, 
                          desc=f'Epoch {iepoch} - Training',
                          position=1, leave=True, ncols=100)
        for k in train_pbar:
            kend = min(k + batch_size, nimg_per_epoch)
            inds = rperm[k:kend]
            imgs, lbls = train._get_batch(inds, data=train_data, labels=train_labels,
                                    files=train_files, labels_files=train_labels_files,
                                    **kwargs)
            diams = np.array([diam_train[i] for i in inds])
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
            # augmentations
            imgi, lbl = random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                            scale_range=scale_range,
                                                            xy=(bsize, bsize))[:2]
            # network and loss optimization
            X = torch.from_numpy(imgi).to(device)
            lbl = torch.from_numpy(lbl).to(device)
            y = net(X)[0]
            loss = train._loss_fn_seg(lbl, y, device)
            if y.shape[1] > 3:
                loss3 = train._loss_fn_class(lbl, y, class_weights=class_weights)
                loss += loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_loss *= len(imgi)

            # keep track of average training loss across epochs
            lavg += train_loss
            nsum += len(imgi)
            # per epoch training loss
            train_losses[iepoch] += train_loss
            
            # Update training progress bar
            train_pbar.set_postfix({
                'loss': f'{train_loss/len(imgi):.4f}',
                'lr': f'{LR[iepoch]:.6f}'
            })
        train_losses[iepoch] /= nimg_per_epoch

        # Validation phase - perform only at specific epochs
        lavgt = 0.
        if (test_data is not None or test_files is not None) and (iepoch % validate_every == 0 or iepoch == n_epochs - 1):
            np.random.seed(42)
            if nimg_test != nimg_test_per_epoch:
                rperm = np.random.choice(np.arange(0, nimg_test),
                                         size=(nimg_test_per_epoch,), p=test_probs)
            else:
                rperm = np.random.permutation(np.arange(0, nimg_test))
                
            # Create progress bar for validation batches
            val_pbar = trange(0, len(rperm), batch_size, 
                            desc=f'Epoch {iepoch} - Validation',
                            position=2, leave=True, ncols=100)
            for ibatch in val_pbar:
                with torch.no_grad():
                    net.eval()
                    inds = rperm[ibatch:ibatch + batch_size]
                    imgs, lbls = train._get_batch(inds, data=test_data,
                                            labels=test_labels, files=test_files,
                                            labels_files=test_labels_files,
                                            **kwargs)
                    diams = np.array([diam_test[i] for i in inds])
                    rsc = diams / net.diam_mean.item() if rescale else np.ones(
                        len(diams), "float32")
                    imgi, lbl = random_rotate_and_resize(
                        imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                        xy=(bsize, bsize))[:2]
                    X = torch.from_numpy(imgi).to(device)
                    lbl = torch.from_numpy(lbl).to(device)
                    y = net(X)[0]
                    loss = train._loss_fn_seg(lbl, y, device)
                    if y.shape[1] > 3:
                        loss3 = train._loss_fn_class(lbl, y, class_weights=class_weights)
                        loss += loss3            
                    test_loss = loss.item()
                    test_loss *= len(imgi)
                    lavgt += test_loss
                    
                    # Update validation progress bar
                    val_pbar.set_postfix({
                        'val_loss': f'{test_loss/len(imgi):.4f}'
                    })
            lavgt /= len(rperm)
            test_losses[iepoch] = lavgt
        else:
            test_losses[iepoch] = test_losses[iepoch-1] if iepoch > 0 else 0

        # Calculate average training loss for logging
        lavg /= nsum
        
        # Clear the line before logging
        print('\033[K', end='')  # Clear the current line
        
        # Log training progress
        if iepoch % validate_every == 0 or iepoch == n_epochs - 1:
            train_logger.info(
                f"Epoch {iepoch} completed - train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )
            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train", lavg, iepoch)
                tb_writer.add_scalar("Loss/valid", lavgt, iepoch)
                tb_writer.add_scalar("Learning Rate", LR[iepoch], iepoch)

        else:
            train_logger.info(
                f"Epoch {iepoch} completed - train_loss={lavg:.4f}, LR={LR[iepoch]:.6f}, time {time.time()-t0:.2f}s"
            )
            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train", lavg, iepoch)
                tb_writer.add_scalar("Learning Rate", LR[iepoch], iepoch)
        
        # Call callback if provided
        if callback is not None:
            callback.on_epoch_end(iepoch, lavg, lavgt if iepoch % validate_every == 0 or iepoch == n_epochs - 1 else test_losses[iepoch], LR[iepoch])
        
        # Reset loss accumulators
        lavg, nsum = 0, 0

        # Save model at specific epochs
        if iepoch == n_epochs - 1 or (iepoch % save_every == 0 and iepoch != 0):
            if save_each and iepoch != n_epochs - 1:
                filename0 = str(filename) + f"_epoch_{iepoch:04d}"
            else:
                filename0 = filename
            train_logger.info(f"saving network parameters to {filename0}")
            net.save_model(filename0)
    
    # Save final model
    net.save_model(filename)
    train_logger.info(f">>> model saved to {filename}")
    return filename, train_losses, test_losses 