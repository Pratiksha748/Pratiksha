import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, figsize=(8, 6), cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt


def plot_sample_predictions(model, generator, class_map, num=6, img_size=224):
    inv_map = {v: k for k, v in class_map.items()} if class_map else None
    imgs = []
    preds = []
    trues = []
    for i in range(len(generator)):
        x, y = generator[i]
        for bx, by in zip(x, y):
            imgs.append(bx)
            preds.append(np.argmax(model.predict(np.expand_dims(bx, 0))[0]))
            trues.append(np.argmax(by))
            if len(imgs) >= num:
                break
        if len(imgs) >= num:
            break

    cols = min(num, 3)
    rows = (num + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for idx in range(num):
        ax = axes[idx]
        ax.imshow(imgs[idx])
        true_label = inv_map[trues[idx]] if inv_map else str(trues[idx])
        pred_label = inv_map[preds[idx]] if inv_map else str(preds[idx])
        ax.set_title(f'True: {true_label}\nPred: {pred_label}')
        ax.axis('off')
    plt.tight_layout()
    return plt
