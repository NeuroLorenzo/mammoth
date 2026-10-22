import logging
from typing import TYPE_CHECKING, Any, Callable, Tuple
import torch
from tqdm.auto import tqdm
import os
import pickle
import numpy as np
import shutil
from datetime import datetime

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset


def mask_classes(outputs: torch.Tensor, dataset: 'ContinualDataset', k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    num_classes = dataset.N_CLASSES
    start_c, end_c = dataset.get_offsets(k)
    outputs[:, :start_c] = -float('inf')
    outputs[:, end_c:num_classes] = -float('inf')


# Type alias for the evaluation function
EvalFn = Callable[
    ['ContinualModel', 'ContinualDataset', bool, bool],
    Any
]


@torch.no_grad()
def evaluate(model: 'ContinualModel', dataset: 'ContinualDataset', last=False, return_loss=False) -> Tuple[list, list]:
    """
    Evaluates the single-class top-1 accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task
        return_loss: a boolean indicating whether to return the loss in addition to the accuracy

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """

    # Genera il timestamp nel formato: YYYY-MM-DD_HH-MM-SS (es. 2026-07-30_23-55-00)
    save_path = "debug_logits"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # shutil.rmtree(save_path)

        
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    loss_fn = dataset.get_loss()
    avg_loss = 0
    tot_seen_samples = 0
    total_len = sum(len(x) for x in dataset.test_loaders) if hasattr(dataset.test_loaders[0], '__len__') else None

    pbar = tqdm(dataset.test_loaders, total=total_len, desc='Evaluating', disable=model.args.non_verbose)
    for k, test_loader in enumerate(dataset.test_loaders):
        all_outputs=[]
        all_labels=[]
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data[0], data[1]
            # print('labels',labels)
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                if model.args.eval_future and k >= model.current_task:
                    outputs = model.future_forward(inputs)
                else:
                    outputs = model(inputs)
                    # print ('labels:',labels)

            if return_loss:
                loss = loss_fn(outputs, labels)
                avg_loss += loss.item()

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1
            pbar.set_postfix({f'acc_task_{k+1}': max(0, correct / total * 100)}, refresh=False)
            pbar.set_description(f"Evaluating Task {k+1}", refresh=False)
            pbar.update(1)
            all_outputs.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        tot_seen_samples += total
        
        if correct > correct_mask_classes and dataset.SETTING == 'class-il':
            logging.warning("Task-IL accuracy is LOWER than Class-IL accuracy. "
                            "This should NEVER happen and probably means there is a bug somewhere. "
                            "Hint: check if the dataloader returns the targets in the correct order.")
        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        filename = f"{save_path}/task_{n_classes}_eval_{k}_logits.pkl"
        with open(filename, 'ab') as f:
            data_to_save = {
                'logits': np.concatenate(all_outputs),
                'labels': np.concatenate(all_labels),
            }
            pickle.dump(data_to_save, f)
    # Salviamo un file per ogni task (k) per non fare confusione
    filename = f"{save_path}/task_{n_classes}_eval.pkl"
    with open(filename, 'ab') as f:
        data_to_save = {
            f'accs':accs,
            f'accs_mask_classes':accs_mask_classes,
        }
        pickle.dump(data_to_save, f)
    pbar.close()

    model.net.train(status)
    if return_loss:
        return accs, accs_mask_classes, avg_loss / tot_seen_samples
    return accs, accs_mask_classes
